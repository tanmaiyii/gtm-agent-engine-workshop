import copy
import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from . import data_service

with patch.dict("os.environ", {"LANGSMITH_TRACING": "false", "OPENAI_API_KEY": "test-key"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ProspectPrivacyTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.cache = patch.object(data_service, "_PROFILES", {})
        self.cache.start()
        self.addCleanup(self.cache.stop)

    def assert_private_fields_absent(self, value):
        serialized = json.dumps(value)
        for key in data_service.SENSITIVE_FIELDS:
            self.assertNotIn(key, serialized)
        for record in data_service.PROSPECTS.values():
            for secret in record["billing_qualification"].values():
                self.assertNotIn(secret, serialized)

    def test_lookup_and_profile_keep_only_task_fields(self):
        for pid, record in data_service.PROSPECTS.items():
            with self.subTest(prospect_id=pid):
                contact = agent.get_prospect.invoke({"prospect_id": pid})["prospect"]
                self.assertEqual(set(contact), {"prospect_id", "name", "email", "disqualified"})
                self.assertEqual(contact["email"], record["email"])
                profile = agent.build_prospect_profile.invoke({"prospect_id": pid})["prospect_profile"]
                self.assertEqual(set(profile), set(data_service.PROFILE_FIELDS))
                self.assertEqual(profile["annual_revenue"], record["annual_revenue"])
                self.assertEqual(profile["tech_stack"], record["tech_stack"])
                self.assert_private_fields_absent(contact)
                self.assert_private_fields_absent(profile)
                self.assert_private_fields_absent(data_service._PROFILES[pid])

    def test_stale_cache_is_cleaned_before_return(self):
        pid = next(iter(data_service.PROSPECTS))
        stale = copy.deepcopy(data_service.PROSPECTS[pid])
        stale["prospect_id"] = pid
        stale["unexpected_field"] = "unneeded"
        stale["account_details"].append({"nested": stale["billing_qualification"]})
        data_service._PROFILES[pid] = stale
        result = data_service.get_profile_from_db(pid)
        self.assert_private_fields_absent(result)
        self.assertNotIn("unexpected_field", result["prospect_profile"])
        self.assert_private_fields_absent(data_service._PROFILES[pid])
        self.assert_private_fields_absent(agent.build_prospect_profile.invoke({"prospect_id": pid}))

    def test_scoring_filters_nested_sensitive_fields(self):
        pid = next(iter(data_service.PROSPECTS))
        profile = copy.deepcopy(data_service.PROSPECTS[pid])
        profile["prospect_id"] = pid
        profile["account_details"].append({"nested": profile["billing_qualification"]})
        offering = next(iter(data_service.OFFERINGS.values()))
        score = agent.ProspectScore(
            score=80, justification="Fit based on revenue and technologies",
            rubric_breakdown=agent.RubricBreakdown(revenue_fit=80, tech_stack_match=80, segment_fit=80),
        )
        llm = Mock()
        llm.invoke.return_value = score
        with patch.object(agent, "_scoring_llm", llm):
            result = agent.score_prospect.invoke({"prospect_profile": profile, "offering": offering})
        self.assertEqual(result["score"], 80)
        messages = llm.invoke.call_args.args[0]
        self.assert_private_fields_absent(messages)
        prompt_profile = json.loads(messages[1]["content"].split("Prospect profile:\n")[1])
        self.assertEqual(prompt_profile["annual_revenue"], profile["annual_revenue"])
        self.assertEqual(prompt_profile["tech_stack"], profile["tech_stack"])
        self.assertIn("billing_qualification", profile)

    def test_reduced_contact_supports_email(self):
        pid = next(iter(data_service.PROSPECTS))
        contact = agent.get_prospect.invoke({"prospect_id": pid})["prospect"]
        runtime = SimpleNamespace(config={"metadata": {"user_id": data_service.REP_IDS[0]["rep_id"]}})
        result = agent.send_prospect_email.func(contact, "Introduction", "Let's connect", runtime)
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["to"], contact["email"])
        self.assertEqual(result["to_name"], contact["name"])
        self.assert_private_fields_absent(result)

    def test_missing_prospect_preserves_not_found_result(self):
        self.assertEqual(agent.get_prospect.invoke({"prospect_id": "missing"}),
                         {"prospect": None, "found": False})
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": "missing"}),
                         {"prospect_profile": None, "found": False})
