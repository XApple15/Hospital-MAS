import json
import re
from typing import Any

from crewai import Agent, Crew, Process, Task, TaskOutput
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent


@CrewBase
class Hospitalmas():
    """
    Hospitalmas crew - orchestrated ontology-driven diagnostic pipeline.

        Agent responsibilities:
            orchestrator       - main manager that coordinates specialist execution
      symptom_extractor  - parse raw user text -> structured symptom list (no tools)
            symp_mapper        - map each symptom -> SYMP URI + numeric ID (GraphDB SYMP search)
      disease_mapper     - query GraphDB per SYMP ID -> disease candidates (SPARQL tool)
      diagnosis_ranker   - rank diseases by co-occurrence -> differential Dx (no tools)
    """

    agents: list[BaseAgent]
    tasks: list[Task]

    runtime_tools: list = []

    @agent
    def orchestrator(self) -> Agent:
        return Agent(
            config=self.agents_config['orchestrator'],
            tools=[],
            verbose=True,
            allow_delegation=True,
            max_iter=20,
            max_retry_limit=2,
        )



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
        symp_tools = [
            t for t in self.runtime_tools
            if getattr(t, "name", "") == "graphdb_symp_search"
        ]
        return Agent(
            config=self.agents_config['symp_mapper'],
            tools=symp_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=18,
            max_retry_limit=2,
        )

    @agent
    def disease_mapper(self) -> Agent:
        sparql_tools = [
            t for t in self.runtime_tools
            if getattr(t, "name", "") == "graphdb_sparql_query"
        ]
        return Agent(
            config=self.agents_config['disease_mapper'],
            tools=sparql_tools,
            verbose=True,
            allow_delegation=False,
            max_iter=20,
            max_retry_limit=2,
        )

    @agent
    def diagnosis_ranker(self) -> Agent:
        return Agent(
            config=self.agents_config['diagnosis_ranker'],
            tools=[],
            verbose=True,
            allow_delegation=False,
            max_iter=3,
            max_retry_limit=1,
        )


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
            context=[self.extract_symptoms_task()]
        )

    @task
    def query_diseases_for_symptoms_task(self) -> Task:
        return Task(
            config=self.tasks_config['query_diseases_for_symptoms_task'],
            agent=self.disease_mapper(),
            context=[self.map_symptoms_to_symp_task()],
        )

    @task
    def rank_diagnoses_task(self) -> Task:
        return Task(
            config=self.tasks_config['rank_diagnoses_task'],
            agent=self.diagnosis_ranker(),
            context=[self.query_diseases_for_symptoms_task()],
        )


    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[
                self.symptom_extractor(),
                self.symp_mapper(),
                self.disease_mapper(),
                self.diagnosis_ranker(),
            ],
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=self.orchestrator(),
            verbose=True,
            max_rpm=5,
        )