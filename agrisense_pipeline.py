# agrisense_pipeline.py

from phase1_crop_recommendation import get_recommended_crops
from phase2_fertilizer import fertilizer_recommendation
from phase3_success_demand import phase3_predict
from phase3_5_price_sell import phase3_5_price_analysis

def run_agrisense(user_input):

    phase1 = get_recommended_crops(**user_input)

    results = {}

    for label in ["primary_crop", "backup_crop"]:
        crop = phase1[label]

        results[label] = {
            "crop_name": crop,
            "fertilizer": fertilizer_recommendation(crop, user_input),
            "success": phase3_predict(crop, user_input),
            "market": phase3_5_price_analysis(crop)
        }

    return results
