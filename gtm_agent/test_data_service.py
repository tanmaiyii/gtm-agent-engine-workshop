import json
import unittest
from copy import deepcopy
from unittest.mock import patch

from . import data_service

with patch.dict("os.environ", {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from . import gtm_agent


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.prospect_id = "test-prospect"
        self.technology = "Kafka"
        self.record = {
            "tech_stack": ["Snowflake"],
            "engagement_history": [],
            "account_details": [],
            "annual_revenue": 80000000,
        }
        for patcher in (
            patch.object(data_service, "PROSPECTS", {self.prospect_id: deepcopy(self.record)}),
            patch.object(data_service, "_PROFILES", {}),
            patch.dict("os.environ", {"LANGSMITH_TRACING": "false"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_update_persists_technology(self):
        result = data_service.update_prospect_info(self.prospect_id, self.technology)

        self.assertEqual(result, {
            "updated": True,
            "found": True,
            "tech_stack": ["Snowflake", "Kafka"],
        })
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), result["tech_stack"])

    def test_update_refreshes_built_profile(self):
        gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        data_service.update_prospect_info(self.prospect_id, self.technology)
        rebuilt = gtm_agent.build_prospect_profile.invoke({"prospect_id": self.prospect_id})

        self.assertTrue(rebuilt["found"])
        self.assertEqual(rebuilt["prospect_profile"]["tech_stack"], ["Snowflake", "Kafka"])
        self.assertEqual(
            data_service.get_profile_from_db(self.prospect_id)["prospect_profile"],
            rebuilt["prospect_profile"],
        )

    def test_score_uses_updated_stack_with_stale_profile(self):
        profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        offering = {
            "required_tech_stack": [self.technology],
            "min_annual_revenue": 50000000,
            "description": "Data platform",
        }

        def score_saved_stack(messages):
            supplied = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
            self.assertIn(self.technology, supplied["tech_stack"])
            return gtm_agent.ProspectScore(
                score=100,
                justification="Has Kafka. Missing required technologies: none.",
                rubric_breakdown=gtm_agent.RubricBreakdown(
                    revenue_fit=100, tech_stack_match=100, segment_fit=100,
                ),
            )

        gtm_agent.update_prospect_info.invoke(
            {"prospect_id": self.prospect_id, "technology": self.technology}
        )
        with patch.object(gtm_agent, "_scoring_llm") as scoring_llm:
            scoring_llm.invoke.side_effect = score_saved_stack
            result = gtm_agent.score_prospect.invoke(
                {"prospect_profile": profile, "offering": offering}
            )

        self.assertEqual(result["rubric_breakdown"]["tech_stack_match"], 100)
        self.assertEqual(result["justification"], "Has Kafka. Missing required technologies: none.")

    def test_update_is_idempotent(self):
        for _ in range(2):
            result = data_service.update_prospect_info(self.prospect_id, "Snowflake")

        self.assertTrue(result["updated"])
        self.assertEqual(result["tech_stack"], ["Snowflake"])
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["Snowflake"])

    def test_missing_prospect_is_unchanged(self):
        result = data_service.update_prospect_info("missing-prospect", self.technology)

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertIsNone(data_service.get_prospect_record("missing-prospect"))
        self.assertEqual(data_service.fetch_tech_stack(self.prospect_id), ["Snowflake"])


if __name__ == "__main__":
    unittest.main()
