import copy
import json
import os
import unittest
from unittest.mock import patch

from langsmith import tracing_context

with patch.dict(os.environ, {"OPENAI_API_KEY": "synthetic-test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import data_service
    from gtm_agent import gtm_agent as agent


SENSITIVE = {
    "billing_qualification": {"reference": "synthetic-billing"},
    "tax_id": "synthetic-tax",
    "date_of_birth": "synthetic-birth",
    "card_on_file": "synthetic-card",
    "credit_check_ref": "synthetic-credit",
}


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.enterContext(tracing_context(enabled=False))
        self.safe_profile = {
            "prospect_id": "TEST-LEAD",
            "name": "Test Prospect",
            "email": "prospect@example.test",
            "disqualified": False,
            "annual_revenue": 1000000,
            "engagement_history": [{"channel": "web"}],
            "account_details": {"segment": "Enterprise"},
            "tech_stack": ["AWS"],
            "custom": {"notes": [{"interest": "analytics"}]},
        }
        self.profile = copy.deepcopy(self.safe_profile)
        self.profile.update(copy.deepcopy(SENSITIVE))
        self.profile["custom"].update(copy.deepcopy(SENSITIVE))
        self.profile["custom"]["notes"][0].update(copy.deepcopy(SENSITIVE))
        self.enterContext(patch.dict(data_service.PROSPECTS, {"TEST-LEAD": self.profile}, clear=True))
        self.enterContext(patch.dict(data_service._PROFILES, {}, clear=True))

    def assert_private(self, value):
        if isinstance(value, dict):
            self.assertFalse(SENSITIVE.keys() & value.keys())
            for item in value.values():
                self.assert_private(item)
        elif isinstance(value, list):
            for item in value:
                self.assert_private(item)
        serialized = json.dumps(value)
        for marker in ("synthetic-billing", "synthetic-tax", "synthetic-birth", "synthetic-card", "synthetic-credit"):
            self.assertNotIn(marker, serialized)

    def test_lookup_allowlists_contact_and_qualification_fields(self):
        result = agent.get_prospect.invoke({"prospect_id": "TEST-LEAD"})
        self.assertEqual(result, {
            "found": True,
            "prospect": {key: self.safe_profile[key] for key in (
                "prospect_id", "name", "email", "disqualified", "annual_revenue",
            )},
        })
        self.assert_private(result)

    def test_new_profile_is_private_before_traced_save(self):
        save_inputs = []
        save = data_service.save_profile_to_db

        def capture_save(prospect_id, profile):
            save_inputs.append(copy.deepcopy(profile))
            return save(prospect_id, profile)

        with patch.object(data_service, "save_profile_to_db", side_effect=capture_save):
            result = agent.build_prospect_profile.invoke({"prospect_id": "TEST-LEAD"})
        self.assertEqual(result, {"prospect_profile": self.safe_profile, "found": True})
        self.assertEqual(save_inputs, [self.safe_profile])
        self.assertEqual(data_service._PROFILES["TEST-LEAD"], self.safe_profile)
        self.assert_private(result)
        self.assert_private(save_inputs)
        self.assertIn("tax_id", self.profile)

    def test_legacy_cached_profile_is_private_at_accessor_and_tool(self):
        data_service._PROFILES["TEST-LEAD"] = self.profile
        cached = data_service.get_profile_from_db("TEST-LEAD")
        result = agent.build_prospect_profile.invoke({"prospect_id": "TEST-LEAD"})
        self.assertEqual(cached, {"prospect_profile": self.safe_profile})
        self.assertEqual(result, {"prospect_profile": self.safe_profile, "found": True})
        self.assert_private(cached)
        self.assert_private(result)

    def test_scoring_sanitizes_nested_profile_and_keeps_record_tech_stack(self):
        profile = copy.deepcopy(self.profile)
        profile["tech_stack"] = ["untrusted stack"]
        offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 500000,
            "description": "Test offering",
        }
        expected = agent.ProspectScore(
            score=90,
            justification="Revenue and AWS fit the offering.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=90, tech_stack_match=100, segment_fit=80,
            ),
        )
        with patch.object(agent, "_scoring_llm") as model:
            model.invoke.return_value = expected
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        messages = model.invoke.call_args.args[0]
        self.assert_private(messages)
        scoring_profile = json.loads(messages[1]["content"].split("Prospect profile:\n", 1)[1])
        self.assertEqual(scoring_profile, self.safe_profile)
        self.assertEqual(result, expected.model_dump())
        self.assertEqual(profile["tech_stack"], ["untrusted stack"])
        self.assertIn("tax_id", profile)

    def test_missing_prospect_and_profile_keep_existing_wrappers(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}), {
            "prospect": None, "found": False,
        })
        self.assertEqual(data_service.get_profile_from_db("missing"), {"prospect_profile": None})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}), {
            "prospect_profile": None, "found": False,
        })
