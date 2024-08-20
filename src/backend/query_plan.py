import re
import json
from pydantic import BaseModel, Field
from typing import List

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
    
class QueryStepExecution(BaseModel):
    search_queries: List[str] | None = Field(
        ...,
        description="The search queries to complete the step",
        min_length=1,
        max_length=3,
    )

def convert_to_query_plan(json_string: str) -> QueryPlan:
    # Parse the JSON string
    steps_data = json.loads(json_string)

    # Convert dependencies that are empty strings to empty lists
    for step in steps_data:
        step['dependencies'] = [int(dep) for dep in step['dependencies'] if dep]

    # Create QueryPlanStep instances
    steps = [QueryPlanStep(**step) for step in steps_data]

    # Create and return QueryPlan instance
    return QueryPlan(steps=steps)

def convert_to_query_step_execution(query_string: str) -> QueryStepExecution:
    # Split the string by line breaks or other delimiter
    queries = [
        query.strip() 
        for query in query_string.split('\n') 
        if query.strip() and re.match(r'^[1-3]\.\s', query.strip())
    ]
    
    # Truncate the list to meet the max_length requirement of 3 items
    if len(queries) > 3:
        queries = queries[:3]
    
    # Create the QueryStepExecution object
    query_step_execution = QueryStepExecution(search_queries=queries)
    
    return query_step_execution

def convert_to_related_queries(query_string: str) -> List[str]:

    # Find the start and end of the list portion
    start_index = query_string.find('[')
    end_index = query_string.rfind(']') + 1

    # Extract the list portion from the input string
    query_string = query_string[start_index:end_index]
       
    # Convert JSON string to Python list
    related_queries = json.loads(query_string)
    
    return related_queries