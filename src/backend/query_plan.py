from pydantic import BaseModel, Field
from typing import List
import json

# Define the QueryPlanStep and QueryPlan classes
class QueryPlanStep(BaseModel):
    id: int = Field(..., description="Unique id of the step")
    step: str
    dependencies: List[int] = Field(
        ...,
        description="List of step ids that this step depends on information from",
        default_factory=list,
    )

class QueryPlan(BaseModel):
    steps: List[QueryPlanStep] = Field(
        ..., description="The steps to complete the query", max_length=4
    )

class QueryPlanStr(BaseModel):
    steps: str

def convert_to_query_plan(query_plan_str: QueryPlanStr) -> QueryPlan:
    # Parse the JSON string from the QueryPlanStr instance
    steps_data = json.loads(query_plan_str.steps)

    # Convert dependencies that are empty strings to empty lists
    for step in steps_data:
        step['dependencies'] = [int(dep) for dep in step['dependencies'] if dep]

    # Create QueryPlanStep instances
    steps = [QueryPlanStep(**step) for step in steps_data]

    # Create and return QueryPlan instance
    return QueryPlan(steps=steps)
