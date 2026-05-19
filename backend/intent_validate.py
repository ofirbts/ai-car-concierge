from backend.database import VehicleSearchFilters, make_exists_in_inventory, model_exists_for_make, search_vehicles
from backend.intent import ExtractedIntent


def normalize_extracted_intent(extracted: ExtractedIntent) -> ExtractedIntent:
    if extracted.make and not make_exists_in_inventory(extracted.make):
        extracted.make = None
        extracted.model = None
    elif extracted.make and extracted.model and not model_exists_for_make(
        extracted.make, extracted.model
    ):
        extracted.model = None

    if extracted.make and extracted.model:
        probe = search_vehicles(
            VehicleSearchFilters(make=extracted.make, model=extracted.model, limit=1)
        )
        if not probe:
            extracted.model = None

    return extracted
