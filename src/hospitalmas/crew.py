import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class Hospitalmas():
    """
    Two-crew human-in-the-loop diagnostic pipeline.

    Phase 1  (diagnostic_crew):
        symptom_extractor → symp_mapper → disease_mapper →
        followup_interviewer
        Runs sequentially (no orchestrator).
        Ranking is computed deterministically in Python after the crew finishes.
        Ends by returning disease_results + follow-up questions.
        main.py pauses here for human input.

    Phase 2  (refine_crew):
        diagnosis_refiner
        Receives initial ranking + answered questions via task description
        and returns the refined differential diagnosis.
        Scoring is also computed deterministically in Python.
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    runtime_tools: list = []
    runtime_log_file: str | None = None

    # ── Agents ───────────────────────────────────────────────────────────────
    # NOTE: orchestrator has been removed. Pipeline uses Process.sequential.

    @agent
    def symptom_extractor(self) -> Agent:
        return Agent(
            config=self.agents_config['symptom_extractor'],
            tools=[],
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=1,
        )

    @agent
    def symp_mapper(self) -> Agent:
        ontology_tools = [
            t for t in self.runtime_tools
            if getattr(t, "name", "") == "graphdb_ontology_query"
        ]
        return Agent(
            config=self.agents_config['symp_mapper'],
            tools=ontology_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=18,
            max_retry_limit=2,
        )

    @agent
    def disease_mapper(self) -> Agent:
        # Give the disease mapper BOTH tools: individual and batch query
        ontology_tools = [
            t for t in self.runtime_tools
            if getattr(t, "name", "") in (
                "graphdb_ontology_query",
                "batch_disease_query",
            )
        ]
        return Agent(
            config=self.agents_config['disease_mapper'],
            tools=ontology_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=20,
            max_retry_limit=2,
        )

    @agent
    def followup_interviewer(self) -> Agent:
        ontology_tools = [
            t for t in self.runtime_tools
            if getattr(t, "name", "") == "graphdb_ontology_query"
        ]
        return Agent(
            config=self.agents_config['followup_interviewer'],
            tools=ontology_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=8,
            max_retry_limit=2,
        )

    @agent
    def diagnosis_refiner(self) -> Agent:
        return Agent(
            config=self.agents_config['diagnosis_refiner'],
            tools=[],
            verbose=True,
            allow_delegation=False,
            max_iter=4,
            max_retry_limit=1,
        )

    # ── Phase 1 tasks ─────────────────────────────────────────────────────────

    @task
    def extract_symptoms_task(self) -> Task:
        return Task(
            config=self.tasks_config['extract_symptoms_task'],
            agent=self.symptom_extractor(),
        )

    @task
    def map_symptoms_to_symp_task(self) -> Task:
        return Task(
            config=self.tasks_config['map_symptoms_to_symp_task'],
            agent=self.symp_mapper(),
            context=[self.extract_symptoms_task()],
        )

    @task
    def query_diseases_for_symptoms_task(self) -> Task:
        return Task(
            config=self.tasks_config['query_diseases_for_symptoms_task'],
            agent=self.disease_mapper(),
            context=[self.map_symptoms_to_symp_task()],
        )

    @task
    def clarify_followup_symptoms_task(self) -> Task:
        return Task(
            config=self.tasks_config['clarify_followup_symptoms_task'],
            agent=self.followup_interviewer(),
            context=[
                self.extract_symptoms_task(),
                self.query_diseases_for_symptoms_task(),
            ],
        )

    # ── Phase 2 task ──────────────────────────────────────────────────────────

    def refine_diagnosis_task_dynamic(self, ranking_json: str, followup_json: str) -> Task:
        """
        Build the refinement task at runtime, injecting the answered follow-up
        data directly into the task description so the refiner agent sees it
        without needing cross-crew context references.
        """
        base_cfg = dict(self.tasks_config['refine_diagnosis_task'])
        base_cfg['description'] = (
            base_cfg.get('description', '').rstrip()
            + "\n\n"
            + "── INITIAL RANKING (JSON) ──────────────────────────────────────\n"
            + ranking_json
            + "\n\n"
            + "── FOLLOW-UP ANSWERS (JSON) ────────────────────────────────────\n"
            + followup_json
        )
        return Task(
            description=base_cfg['description'],
            expected_output=base_cfg.get('expected_output', ''),
            agent=self.diagnosis_refiner(),
        )

    # ── Crew factories ────────────────────────────────────────────────────────

    @crew
    def crew(self) -> Crew:
        """Default crew alias — runs Phase 1 only (for train/test/replay)."""
        return self.diagnostic_crew()

    def diagnostic_crew(self) -> Crew:
        """
        Phase 1: diagnostic pipeline (sequential, no orchestrator).

        Runs: symptom extraction → SYMP mapping → disease mapping →
              follow-up question generation.

        NOTE: The rank_diagnoses_task has been removed from the crew.
        Ranking is now computed deterministically in Python by main.py
        after this crew finishes, using scoring.compute_ranking().
        The followup_interviewer receives disease_results directly
        and the ranking is injected into its context by the pipeline.
        """
        return Crew(
            agents=[
                self.symptom_extractor(),
                self.symp_mapper(),
                self.disease_mapper(),
                self.followup_interviewer(),
            ],
            tasks=[
                self.extract_symptoms_task(),
                self.map_symptoms_to_symp_task(),
                self.query_diseases_for_symptoms_task(),
                self.clarify_followup_symptoms_task(),
            ],
            process=Process.sequential,
            verbose=True,
            max_rpm=5,
            output_log_file=self.runtime_log_file,
        )

    def refine_crew(self, ranking_json: str, followup_json: str) -> Crew:
        """
        Phase 2: refine diagnosis using patient answers.
        ranking_json and followup_json are serialised strings injected into
        the task description so the refiner has full context.
        """
        return Crew(
            agents=[self.diagnosis_refiner()],
            tasks=[self.refine_diagnosis_task_dynamic(ranking_json, followup_json)],
            process=Process.sequential,
            verbose=True,
            max_rpm=5,
            output_log_file=self.runtime_log_file,
        )