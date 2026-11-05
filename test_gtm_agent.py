import importlib
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

with patch("langchain_openai.ChatOpenAI"), patch("deepagents.create_deep_agent"):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class GTMAgentTests(unittest.TestCase):
    def setUp(self):
        self.tracing = patch.dict("os.environ", {"LANGSMITH_TRACING": "false"})
        self.tracing.start()
        self.addCleanup(self.tracing.stop)
        self.profiles = patch.object(agent.data_service, "_PROFILES", {})
        self.profiles.start()
        self.addCleanup(self.profiles.stop)

    def test_scoring_ids_preserve_scoring_inputs_and_result(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id = next(iter(agent.data_service.OFFERINGS))
        profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
        expected_profile = {
            **profile,
            "tech_stack": agent.data_service.fetch_tech_stack(prospect_id),
        }
        agent.data_service._PROFILES.clear()
        score = {"score": 75, "justification": "Example fit"}
        with patch.object(agent, "_scoring_llm") as scoring:
            scoring.invoke.return_value.model_dump.return_value = score
            result = agent.score_prospect.invoke({
                "prospect_id": prospect_id, "offering_id": offering_id,
            })
        self.assertEqual(result, score)
        messages = scoring.invoke.call_args.args[0]
        self.assertEqual(messages, [
            {"role": "system", "content": agent.SCORING_PROMPT},
            {"role": "user", "content": (
                "Offering:\n" + json.dumps(agent.data_service.OFFERINGS[offering_id], indent=2)
                + "\n\nProspect profile:\n" + json.dumps(expected_profile, indent=2)
            )},
        ])
        self.assertEqual(agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"], profile)

    def test_scoring_uses_current_stack_with_cached_profile(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id = next(iter(agent.data_service.OFFERINGS))
        profile = agent.build_prospect_profile.invoke({"prospect_id": prospect_id})["prospect_profile"]
        profile["tech_stack"] = ["stale"]
        with patch.object(agent, "_scoring_llm") as scoring:
            agent.score_prospect.invoke({"prospect_id": prospect_id, "offering_id": offering_id})
        payload = scoring.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(payload.split("\n\nProspect profile:\n")[1])
        self.assertEqual(scored_profile["tech_stack"], agent.data_service.fetch_tech_stack(prospect_id))
        self.assertEqual(profile["tech_stack"], ["stale"])

    def test_scoring_missing_records_does_not_invoke_model(self):
        prospect_id = next(iter(agent.data_service.PROSPECTS))
        offering_id = next(iter(agent.data_service.OFFERINGS))
        with patch.object(agent, "_scoring_llm") as scoring:
            for pid, oid in [(prospect_id, "missing"), ("missing", offering_id)]:
                with self.subTest(prospect_id=pid, offering_id=oid):
                    result = agent.score_prospect.invoke({"prospect_id": pid, "offering_id": oid})
                    self.assertIsNone(result["score"])
                    self.assertIn("error", result)
            scoring.invoke.assert_not_called()

    def test_rep_context_and_email_sender_match(self):
        rep = agent.REP_IDS[0]
        graph = Mock()
        graph.invoke.return_value = {"messages": [SimpleNamespace(content="Sent")]}
        with patch.object(agent, "gtm_agent", graph):
            result = agent.run_agent("Email the prospect", user_id=rep["rep_id"])
        messages = graph.invoke.call_args.args[0]["messages"]
        self.assertEqual(messages[0]["role"], "system")
        self.assertIn(rep["name"], messages[0]["content"])
        self.assertIn(rep["email"], messages[0]["content"])
        self.assertEqual(messages[1], {"role": "user", "content": "Email the prospect"})
        runtime = SimpleNamespace(config=graph.invoke.call_args.kwargs["config"])
        sent = agent.send_prospect_email.func(
            {"name": "Prospect", "email": "prospect@example.com"},
            "Hello", f"Regards, {rep['name']}", runtime,
        )
        self.assertEqual(sent["from"], rep["email"])
        self.assertEqual(sent["from_name"], rep["name"])
        self.assertEqual(sent["body"], f"Regards, {rep['name']}")
        self.assertEqual(result["reply"], "Sent")

    def test_unknown_rep_omits_context_and_keeps_fallback(self):
        graph = Mock()
        graph.invoke.return_value = {"messages": [SimpleNamespace(content="Done")]}
        with patch.object(agent, "gtm_agent", graph):
            agent.run_agent("Score a prospect", user_id="missing")
        self.assertEqual(graph.invoke.call_args.args[0]["messages"], [
            {"role": "user", "content": "Score a prospect"},
        ])
        runtime = SimpleNamespace(config=graph.invoke.call_args.kwargs["config"])
        self.assertEqual(agent.get_current_rep.func(runtime), {"rep": None, "found": False})


if __name__ == "__main__":
    unittest.main()
