import json
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"LANGSMITH_TRACING": "false"}), \
        patch("langchain_openai.ChatOpenAI"), \
        patch("deepagents.create_deep_agent"):
    from gtm_agent import gtm_agent as agent


class ScoringTests(unittest.TestCase):
    def setUp(self):
        self.cache = patch.dict(agent.data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.scorer = patch.object(agent, "_scoring_llm")
        self.llm = self.scorer.start()
        self.addCleanup(self.scorer.stop)
        self.expected = {
            "score": 75.0,
            "max_score": 100,
            "justification": "Good fit with a missing required technology.",
            "rubric_breakdown": {
                "revenue_fit": 90.0,
                "tech_stack_match": 60.0,
                "segment_fit": 75.0,
                "component_max": 100,
            },
        }
        self.llm.invoke.return_value = agent.ProspectScore(**self.expected)

    def test_scores_by_ids_with_the_existing_scoring_payload(self):
        profile = agent.build_prospect_profile.invoke(
            {"prospect_id": "LEAD-71001"}
        )["prospect_profile"]
        agent.data_service._PROFILES.clear()
        result = agent.score_prospect.invoke(
            {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
        )
        self.assertEqual(result, self.expected)
        messages = self.llm.invoke.call_args.args[0]
        self.assertEqual(messages, [
            {"role": "system", "content": agent.SCORING_PROMPT},
            {"role": "user", "content": (
                "Offering:\n" + json.dumps(
                    agent.data_service.get_offering("OFFER-10005"), indent=2
                ) + "\n\nProspect profile:\n" + json.dumps(profile, indent=2)
            )},
        ])
        self.assertEqual(agent.build_prospect_profile.invoke(
            {"prospect_id": "LEAD-71001"}
        ), {"prospect_profile": profile, "found": True})

    def test_refreshes_cached_tech_stack_without_changing_cached_profile(self):
        profile = agent.build_prospect_profile.invoke(
            {"prospect_id": "LEAD-71001"}
        )["prospect_profile"]
        saved_stack = list(profile["tech_stack"])
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["fresh"]):
            agent.score_prospect.invoke(
                {"prospect_id": "LEAD-71001", "offering_id": "OFFER-10005"}
            )
        payload = self.llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(payload.split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile["tech_stack"], ["fresh"])
        self.assertEqual(profile["tech_stack"], saved_stack)

    def test_rejects_missing_or_incomplete_offerings(self):
        for offering in (None, {}, {"description": "Incomplete"}):
            with self.subTest(offering=offering), patch.object(
                agent.data_service, "get_offering", return_value=offering
            ):
                self.assertEqual(agent.score_prospect.invoke({
                    "prospect_id": "LEAD-71001", "offering_id": "missing"
                }), {"score": None, "error": "Cannot score without a valid offering."})
        self.llm.invoke.assert_not_called()

    def test_rejects_missing_prospect(self):
        self.assertEqual(agent.score_prospect.invoke({
            "prospect_id": "missing", "offering_id": "OFFER-10005"
        }), {"score": None, "error": "Prospect not found."})
        self.llm.invoke.assert_not_called()
        self.assertEqual(agent.build_prospect_profile.invoke(
            {"prospect_id": "missing"}
        ), {"prospect_profile": None, "found": False})


if __name__ == "__main__":
    unittest.main()
