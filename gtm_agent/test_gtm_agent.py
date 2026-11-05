import importlib
import json
import unittest
from unittest.mock import Mock, patch

from langsmith import tracing_context

from gtm_agent import data_service


class ProspectPrivacyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
            cls.agent = importlib.import_module("gtm_agent.gtm_agent")

    def setUp(self):
        self.cache = patch.dict(data_service._PROFILES, {}, clear=True)
        self.cache.start()
        self.addCleanup(self.cache.stop)
        self.tracing = tracing_context(enabled=False)
        self.tracing.__enter__()
        self.addCleanup(self.tracing.__exit__, None, None, None)
        self.prospect_id = "LEAD-12853"
        self.record = data_service.get_prospect_record(self.prospect_id)
        self.safe_record = {
            k: v for k, v in self.record.items() if k != "billing_qualification"
        }

    def assert_no_sensitive_data(self, value):
        serialized = json.dumps(value)
        sensitive = self.record["billing_qualification"]
        for text in ("billing_qualification", *sensitive.keys(), *sensitive.values()):
            self.assertNotIn(text, serialized)

    def test_get_prospect_excludes_billing(self):
        result = self.agent.get_prospect.invoke({"prospect_id": self.prospect_id})
        self.assert_no_sensitive_data(result)
        expected = {
            k: v for k, v in self.safe_record.items()
            if k not in ("engagement_history", "account_details", "tech_stack")
        }
        self.assertEqual(result, {
            "prospect": {"prospect_id": self.prospect_id, **expected}, "found": True,
        })

    def test_fresh_and_cached_profiles_exclude_billing(self):
        expected = {"prospect_id": self.prospect_id, **self.safe_record}
        for cached in (False, True):
            with self.subTest(cached=cached):
                if cached:
                    data_service.save_profile_to_db(
                        self.prospect_id, {"prospect_id": self.prospect_id, **self.record}
                    )
                result = self.agent.build_prospect_profile.invoke({
                    "prospect_id": self.prospect_id,
                })
                stored = data_service.get_profile_from_db(self.prospect_id)
                self.assert_no_sensitive_data(result)
                self.assert_no_sensitive_data(stored)
                self.assertEqual(result, {"prospect_profile": expected, "found": True})
                self.assertEqual(stored, {"prospect_profile": expected})

    def test_scoring_excludes_billing_from_model_input(self):
        profile = {"prospect_id": self.prospect_id, **self.record, "tech_stack": []}
        offering = data_service.get_offering("OFFER-10001")
        score = {"score": 75, "justification": "Good fit"}
        llm = Mock()
        llm.invoke.return_value.model_dump.return_value = score
        with patch.object(self.agent, "_scoring_llm", llm):
            result = self.agent.score_prospect.invoke({
                "prospect_profile": profile, "offering": offering,
            })
        messages = llm.invoke.call_args.args[0]
        self.assert_no_sensitive_data(messages)
        prompt_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(prompt_profile, {"prospect_id": self.prospect_id, **self.safe_record})
        self.assertEqual(result, score)
        self.assertIn("billing_qualification", profile)


if __name__ == "__main__":
    unittest.main()
