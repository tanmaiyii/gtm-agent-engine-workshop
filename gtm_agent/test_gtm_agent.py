import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key", "LANGSMITH_TRACING": "false"}):
    from gtm_agent import gtm_agent as agent


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_mrossi"}})

    def send(self, prospect):
        return agent.send_prospect_email.func(
            prospect, "Demo invitation", "Please book a demo.", self.runtime
        )

    def test_disqualified_record_blocks_even_if_contact_says_qualified(self):
        prospect = {"prospect_id": "test-lead", "email": "lead@example.com", "disqualified": False}
        with patch.object(agent.data_service, "get_prospect_record", return_value={"disqualified": "yes"}):
            with patch.object(agent.uuid, "uuid4", side_effect=AssertionError("Message ID created")):
                result = self.send(prospect)
        self.assertEqual(result, {"status": "blocked", "reason": "disqualified", "to": prospect["email"]})
        self.assertNotIn("message_id", result)

    def test_lead_50001_is_blocked(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-50001"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result, {"status": "blocked", "reason": "disqualified", "to": prospect["email"]})
        self.assertNotIn("message_id", result)

    def test_qualified_prospect_is_sent(self):
        prospect = agent.get_prospect.invoke({"prospect_id": "LEAD-12853"})["prospect"]
        result = self.send(prospect)
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"].startswith("msg-"))
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["subject"], "Demo invitation")
        self.assertEqual(result["body"], "Please book a demo.")
        self.assertEqual(result["from"], agent.data_service.get_rep("rep_mrossi")["email"])

    def test_missing_record_preserves_sending(self):
        for prospect_id in ("missing-lead", None):
            with self.subTest(prospect_id=prospect_id):
                result = self.send({"prospect_id": prospect_id, "email": "lead@example.com"})
                self.assertEqual(result["status"], "sent")
                self.assertTrue(result["message_id"].startswith("msg-"))
                self.assertEqual(result["to"], "lead@example.com")

    def test_missing_email_preserves_failure(self):
        self.assertEqual(self.send({"prospect_id": "LEAD-12853"}), {
            "status": "failed", "error": "Prospect record has no email address."
        })


if __name__ == "__main__":
    unittest.main()
