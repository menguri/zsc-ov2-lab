import os
from datetime import datetime
from pathlib import Path
from typing import List
import jax


def get_recipe_identifier(ingredients: List[int]) -> int:
    """
    Get the identifier for a recipe given the ingredients.
    """
    return f"{ingredients[0]}_{ingredients[1]}_{ingredients[2]}"
