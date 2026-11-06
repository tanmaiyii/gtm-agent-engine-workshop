import importlib
import json
import os
import re
import unittest
import uuid
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "offline-test",
                "LANGSMITH_TRACING": "false",
            },
        )
        cls.env.start()
        cls.agent = importlib.import_module("gtm_agent.gtm_agent")

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def test_invocation_config_and_returned_messages(self):
        messages = [HumanMessage(content="Score a prospect"), AIMessage(content="Done")]
        graph = Mock()
        graph.invoke.return_value = {"messages": messages}
        with patch.object(self.agent, "gtm_agent", graph):
            result = self.agent.run_agent(
                "Score a prospect",
                user_id="rep_jchen",
                environment="test",
                thread_id="thread-123",
            )
        (inputs,) = graph.invoke.call_args.args
        config = graph.invoke.call_args.kwargs["config"]
        self.assertEqual(
            inputs, {"messages": [{"role": "user", "content": "Score a prospect"}]}
        )
        self.assertEqual(
            config,
            {
                "run_name": "GTM Assistant",
                "run_id": uuid.UUID(result["run_id"]),
                "metadata": {
                    "thread_id": "thread-123",
                    "user_id": "rep_jchen",
                    "environment": "test",
                    "request_intent": "score_prospect",
                },
            },
        )
        self.assertEqual(result["reply"], "Done")
        self.assertIs(result["messages"], messages)

    def test_example_workflows_with_only_gtm_prompt_and_tools(self):
        from run import EXAMPLES

        expected_tools = {
            "lookup_offering",
            "build_prospect_profile",
            "get_prospect",
            "send_prospect_email",
            "score_prospect",
            "update_prospect_info",
            "get_current_rep",
        }
        score = self.agent.ProspectScore(
            score=75,
            justification="Offline score",
            rubric_breakdown=self.agent.RubricBreakdown(
                revenue_fit=80,
                tech_stack_match=70,
                segment_fit=75,
            ),
        )
        for rep_id, request in EXAMPLES:
            with self.subTest(request=request):
                self.check_workflow(rep_id, request, expected_tools, score)

    def check_workflow(self, rep_id, request, expected_tools, score):
        prospect_id = re.search(r"LEAD-\d+", request).group()
        offering_match = re.search(r"OFFER-\d+", request)
        sequence = ["get_current_rep"]
        if offering_match:
            if "add " in request.lower() or "update" in request.lower():
                sequence.append("update_prospect_info")
            sequence += [
                "lookup_offering",
                "build_prospect_profile",
                "score_prospect",
            ]
        else:
            sequence += ["get_prospect", "send_prospect_email"]
        observed = []

        def generate(model, messages, **kwargs):
            self.assertEqual(
                [m.content for m in messages if isinstance(m, SystemMessage)],
                [self.agent.SYSTEM_PROMPT],
            )
            self.assertEqual(
                {t["function"]["name"] for t in kwargs["tools"]}, expected_tools
            )
            results = {
                m.name: json.loads(m.content)
                for m in messages
                if isinstance(m, ToolMessage)
            }
            if len(observed) == len(sequence):
                if offering_match:
                    self.assertEqual(results["score_prospect"]["score"], 75)
                    reply = "Prospect score: 75/100."
                else:
                    sent = results["send_prospect_email"]
                    self.assertEqual(sent["status"], "sent")
                    self.assertEqual(
                        sent["from"],
                        self.agent.data_service.get_rep(rep_id)["email"],
                    )
                    self.assertEqual(
                        sent["to"],
                        self.agent.data_service.get_prospect_record(prospect_id)[
                            "email"
                        ],
                    )
                    reply = "Email sent."
                message = AIMessage(content=reply)
            else:
                name = sequence[len(observed)]
                args = {}
                if name in (
                    "get_prospect",
                    "build_prospect_profile",
                    "update_prospect_info",
                ):
                    args = {"prospect_id": prospect_id}
                if name == "update_prospect_info":
                    args["technology"] = next(
                        t for t in ("Kafka", "Terraform", "Okta") if t in request
                    )
                if name == "lookup_offering":
                    args = {"offering_id": offering_match.group()}
                if name == "score_prospect":
                    args = {
                        "prospect_profile": results["build_prospect_profile"][
                            "prospect_profile"
                        ],
                        "offering": results["lookup_offering"]["offering"],
                    }
                if name == "send_prospect_email":
                    args = {
                        "prospect": results["get_prospect"]["prospect"],
                        "subject": "Meeting",
                        "body": request,
                    }
                observed.append(name)
                message = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": name,
                            "args": args,
                            "id": str(len(observed)),
                        }
                    ],
                )
            return ChatResult(generations=[ChatGeneration(message=message)])

        with (
            patch.object(ChatOpenAI, "_generate", generate),
            patch.object(
                self.agent,
                "_scoring_llm",
                Mock(invoke=Mock(return_value=score)),
            ),
        ):
            result = self.agent.run_agent(request, user_id=rep_id)
        self.assertEqual(observed, sequence)
        self.assertEqual(
            result["reply"],
            "Prospect score: 75/100." if offering_match else "Email sent.",
        )
        self.assertEqual(
            [m.name for m in result["messages"] if isinstance(m, ToolMessage)],
            sequence,
        )


if __name__ == "__main__":
    unittest.main()
