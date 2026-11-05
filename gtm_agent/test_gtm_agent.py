import unittest
from types import SimpleNamespace

from gtm_agent.gtm_agent import get_prospect, send_prospect_email


class SendProspectEmailTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(config={"metadata": {"user_id": "rep_amills"}})

    def test_disqualified_prospect_is_blocked(self):
        for prospect_id in ("LEAD-50001", "LEAD-50002", "LEAD-50003", "LEAD-50004", "LEAD-50005"):
            with self.subTest(prospect_id=prospect_id):
                prospect = get_prospect.invoke({"prospect_id": prospect_id})["prospect"]
                self.assertTrue(prospect["disqualified"])
                result = send_prospect_email.func(
                    prospect, "Book a demo", "Please book a demo.", self.runtime
                )
                self.assertEqual(result, {
                    "status": "blocked",
                    "error": "Prospect is disqualified; email not sent.",
                })

    def test_source_record_blocks_send_when_caller_omits_or_changes_flag(self):
        for fields in ({}, {"disqualified": False}):
            with self.subTest(fields=fields):
                prospect = {
                    "prospect_id": "LEAD-50001",
                    "email": "prospect@example.com",
                    **fields,
                }
                result = send_prospect_email.func(
                    prospect, "Book a demo", "Please book a demo.", self.runtime
                )
                self.assertEqual(result["status"], "blocked")
                self.assertNotIn("message_id", result)

    def test_qualified_prospect_is_sent(self):
        prospect = get_prospect.invoke({"prospect_id": "LEAD-70001"})["prospect"]
        self.assertFalse(prospect["disqualified"])
        result = send_prospect_email.func(
            prospect, "Book a demo", "Please book a demo.", self.runtime
        )
        self.assertEqual(result["status"], "sent")
        self.assertTrue(result["message_id"])
        self.assertEqual(result["to"], prospect["email"])
        self.assertEqual(result["from"], "ava.mills@northpoint.com")
        self.assertEqual(result["subject"], "Book a demo")
        self.assertEqual(result["body"], "Please book a demo.")
        self.assertNotIn("billing_qualification", result)


if __name__ == "__main__":
    unittest.main()
