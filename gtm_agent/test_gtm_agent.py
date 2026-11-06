"""Offline checks for representative identity in the agent workflow."""

import json
import os
import re
import unittest
from unittest.mock import patch

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service
from gtm_agent import gtm_agent as agent
from run import EXAMPLES


class WorkflowModel(BaseChatModel):
    prompts: list = Field(default_factory=list)

    @property
    def _llm_type(self):
        return "test-workflow"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.prompts.append(messages)
        request = next(m.content for m in messages if isinstance(m, HumanMessage))
        results = {
            m.name: json.loads(m.content)
            for m in messages if isinstance(m, ToolMessage)
        }
        prompt = messages[0].text
        rep = next((r for r in data_service.REP_IDS
                    if r["name"] in prompt and r["email"] in prompt), None)
        calls = []
        if rep is None and "get_current_rep" not in results:
            calls = [("get_current_rep", {})]
        elif "email" in request.lower():
            rep = rep or results["get_current_rep"]["rep"]
            if "get_prospect" not in results:
                calls = [("get_prospect", {"prospect_id": re.search(r"LEAD-\d+", request)[0]})]
            elif "send_prospect_email" not in results:
                calls = [("send_prospect_email", {
                    "prospect": results["get_prospect"]["prospect"],
                    "subject": "Follow up",
                    "body": f"Let's schedule a call.\n\n{rep['name']}\n{rep['email']}",
                })]
        elif "add" in request.lower() and "update_prospect_info" not in results:
            calls = [("update_prospect_info", {"prospect_id": "LEAD-71001", "technology": "Kafka"})]
        elif "build_prospect_profile" not in results:
            calls = [
                ("build_prospect_profile", {"prospect_id": "LEAD-71001"}),
                ("lookup_offering", {"offering_id": "OFFER-10005"}),
            ]
        elif "score_prospect" not in results:
            calls = [("score_prospect", {
                "prospect_profile": results["build_prospect_profile"]["prospect_profile"],
                "offering": results["lookup_offering"]["offering"],
            })]
        response = AIMessage(content="" if calls else "Done", tool_calls=[
            {"name": name, "args": args, "id": f"call-{len(self.prompts)}-{i}"}
            for i, (name, args) in enumerate(calls)
        ])
        return ChatResult(generations=[ChatGeneration(message=response)])


class RepresentativeWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.model = WorkflowModel()
        model_patch = patch.object(type(agent.agent_model), "_generate", side_effect=self.model._generate)
        model_patch.start()
        self.addCleanup(model_patch.stop)

    def test_example_emails_use_configured_sender_in_three_model_calls(self):
        for user_id, request in EXAMPLES:
            if "email" not in request.lower():
                continue
            with self.subTest(user_id=user_id):
                self.model.prompts.clear()
                result = agent.run_agent(request, user_id=user_id)
                rep = data_service.get_rep(user_id)
                sent = next(json.loads(m.content) for m in result["messages"]
                            if isinstance(m, ToolMessage) and m.name == "send_prospect_email")
                self.assertEqual(sent["status"], "sent")
                self.assertEqual(sent["from"], rep["email"])
                self.assertEqual(sent["from_name"], rep["name"])
                self.assertIn(rep["name"], sent["body"])
                self.assertIn(rep["email"], sent["body"])
                self.assertEqual(len(self.model.prompts), 3)
                self.assertIn(rep["name"], self.model.prompts[0][0].text)
                self.assertIn(rep["email"], self.model.prompts[0][0].text)
                self.assertFalse(any(isinstance(m, ToolMessage) and m.name == "get_current_rep"
                                     for m in result["messages"]))

    def test_scoring_and_updates_complete_without_identity_tool(self):
        for request, expected_calls in [
            ("Score LEAD-71001 against OFFER-10005.", 3),
            ("First add Kafka to LEAD-71001's tech stack, then score them against OFFER-10005.", 4),
        ]:
            with self.subTest(request=request):
                self.model.prompts.clear()
                score = {"score": 80, "justification": "Fit", "rubric_breakdown": {
                    "revenue_fit": 80, "tech_stack_match": 80, "segment_fit": 80,
                }}
                with patch.object(agent, "_scoring_llm") as scoring_model:
                    scoring_model.invoke.return_value = agent.ProspectScore(**score)
                    result = agent.run_agent(request, user_id="rep_jchen")
                tools = [m.name for m in result["messages"] if isinstance(m, ToolMessage)]
                self.assertIn("score_prospect", tools)
                if "add" in request:
                    self.assertIn("update_prospect_info", tools)
                self.assertNotIn("get_current_rep", tools)
                self.assertEqual(len(self.model.prompts), expected_calls)


if __name__ == "__main__":
    unittest.main()
