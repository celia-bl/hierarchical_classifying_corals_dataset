from codecarbon import EmissionsTracker
import os

# Initialize tracker
tracker = EmissionsTracker(allow_multiple_runs=True)

class EmissionsData:
    def __init__(self, energy_consumed: float, emissions: float):
        self.energy_consumed = energy_consumed
        self.emissions = emissions

def clean_emissions_data(emissions_data):
    """Remove unwanted fields from emissions data"""
    data_dict = emissions_data.__dict__
    fields_to_remove = ['timestamp', 'project_name', 'experiment_id', 'latitude', 'longitude']
    return {k: v for k, v in data_dict.items() if k not in fields_to_remove}

def get_space_info():
    """Get the space username and URL from environment variables"""
    space_name = os.getenv("SPACE_ID", "")
    if space_name:
        try:
            username = space_name.split("/")[0]
            space_url = f"https://huggingface.co/spaces/{space_name}"
            return username, space_url
        except Exception as e:
            print(f"Error getting space info: {e}")
    return "local-user", "local-development"