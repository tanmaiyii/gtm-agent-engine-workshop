import os
import unittest
from unittest.mock import patch

from dotenv import load_dotenv

load_dotenv(override=True)


class PlainTextEmailTests(unittest.TestCase):
    def test_attachment_mention_does_not_block_email(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}):
            from gtm_agent.gtm_agent import send_prospect_email

        body = "Please reply without attachments; a link is fine."
        result = send_prospect_email.func(
            prospect={"name": "Prospect", "email": "prospect@example.com"},
            subject="Follow-up",
            body=body,
            runtime=None,
            from_rep={"name": "Rep", "email": "rep@example.com"},
        )
        self.assertEqual(result["status"], "sent")
        self.assertEqual(result["body"], body)
        self.assertEqual(result["to"], "prospect@example.com")


@unittest.skipUnless(
    os.environ.get("OPENAI_API_KEY"),
    "Live agent tests require model provider credentials.",
)
class DocumentEmailTests(unittest.TestCase):
    def test_pricing_deck_without_link(self):
        from gtm_agent.gtm_agent import run_agent

        result = run_agent(
            "Email LEAD-50003 to send over our pricing deck.",
            user_id="rep_sbrown",
        )
        reply = result["reply"]
        self.assertRegex(
            reply.lower(),
            r"(?:cannot|can't|unable to|don't have|do not have)[^.]*"
            r"(?:attach|fetch|access|retrieve)",
        )
        self.assertRegex(reply.lower(), r"(?:provide|share|send|supply)[^.]*link")
        self.assertNotRegex(
            reply.lower(),
            r"(?:i have|i've|successfully) (?:sent|emailed|attached|included)|"
            r"(?:deck|document) (?:has been|was|is now) (?:sent|attached|included)",
        )
        for message in result["messages"]:
            for call in getattr(message, "tool_calls", []):
                self.assertNotEqual(call["name"], "send_prospect_email")

    def test_pricing_deck_with_link(self):
        from gtm_agent.gtm_agent import run_agent

        link = "https://example.com/pricing-deck.pdf"
        result = run_agent(
            f"Email LEAD-50003 our pricing deck using this link: {link}",
            user_id="rep_sbrown",
        )
        sent = [
            call for message in result["messages"]
            for call in getattr(message, "tool_calls", [])
            if call["name"] == "send_prospect_email"
        ]
        self.assertTrue(sent)
        for call in sent:
            self.assertIn(link, call["args"]["body"])
            self.assertNotRegex(
                call["args"]["body"].lower(),
                r"(?:attached is|please find attached|i have attached|i've attached)",
            )


if __name__ == "__main__":
    unittest.main()
