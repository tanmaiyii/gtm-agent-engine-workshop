import json
import unittest
from copy import deepcopy
from unittest.mock import Mock, patch

from gtm_agent import data_service

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "LEAD-71001"
        self.enterContext(patch.object(data_service, "PROSPECTS", deepcopy(data_service.PROSPECTS)))
        self.enterContext(patch.object(data_service, "_PROFILES", {}))
        self.enterContext(patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}))

    def test_update_persists_technology_and_refreshes_cached_profile(self):
        original = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertNotIn("Kafka", original["prospect_profile"]["tech_stack"])

        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Kafka", result["tech_stack"])
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        self.assertIn("Kafka", rebuilt["prospect_profile"]["tech_stack"])

    def test_scoring_receives_saved_update_even_with_old_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})
        data_service.update_prospect_info(self.prospect_id, "Kafka")
        scoring_llm = Mock()
        with patch.object(gtm_agent, "_scoring_llm", scoring_llm):
            gtm_agent.score_prospect.invoke({
                "prospect_profile": profile["prospect_profile"],
                "offering": data_service.get_offering("OFFER-10001"),
            })

        messages = scoring_llm.invoke.call_args.args[0]
        scored_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertIn("Kafka", scored_profile["tech_stack"])

    def test_already_present_technology_is_not_duplicated(self):
        original = list(data_service.fetch_tech_stack(self.prospect_id))
        technology = original[0]

        result = data_service.update_prospect_info(self.prospect_id, technology)

        self.assertFalse(result["updated"])
        self.assertTrue(result["found"])
        self.assertTrue(result["already_present"])
        self.assertEqual(original, result["tech_stack"])
        self.assertEqual(original, data_service.fetch_tech_stack(self.prospect_id))

    def test_missing_prospect_is_not_updated(self):
        self.assertEqual(
            {"updated": False, "found": False},
            data_service.update_prospect_info("unknown", "Kafka"),
        )


if __name__ == "__main__":
    unittest.main()
