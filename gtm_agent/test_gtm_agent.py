import importlib
import json
import os
import unittest
from unittest.mock import patch

with patch.dict(os.environ, {"OPENAI_API_KEY": "test", "LANGSMITH_TRACING": "false"}):
    agent = importlib.import_module("gtm_agent.gtm_agent")


class ScoreProspectTests(unittest.TestCase):
    def setUp(self):
        self.offering = {
            "required_tech_stack": ["AWS"],
            "min_annual_revenue": 1000000,
            "description": "Cloud platform",
        }

    def test_weighted_total_and_response_shape(self):
        cases = [
            ((100, 75, 100), 90.0),
            ((100, 70, 100), 88.0),
            ((81.23, 62.34, 43.45), 66.1),
            ((0, 0, 0), 0.0),
            ((100, 100, 100), 100.0),
        ]
        for components, expected in cases:
            with self.subTest(components=components):
                breakdown = agent.RubricBreakdown(
                    revenue_fit=components[0],
                    tech_stack_match=components[1],
                    segment_fit=components[2],
                )
                assessment = agent.ScoringAssessment(
                    justification="AWS is present.", rubric_breakdown=breakdown,
                )
                with patch.object(agent, "_scoring_llm") as llm:
                    llm.invoke.return_value = assessment
                    results = [agent.score_prospect.func({}, self.offering) for _ in range(2)]
                self.assertEqual(results[0], results[1])
                self.assertEqual(results[0], {
                    "score": expected,
                    "max_score": 100,
                    "justification": "AWS is present.",
                    "rubric_breakdown": {
                        "revenue_fit": components[0],
                        "tech_stack_match": components[1],
                        "segment_fit": components[2],
                        "component_max": 100,
                    },
                })

    def test_invalid_offering(self):
        for offering in (None, {}, {**self.offering, "required_tech_stack": []}):
            with self.subTest(offering=offering):
                self.assertEqual(agent.score_prospect.func({}, offering), {
                    "score": None, "error": "Cannot score without a valid offering.",
                })

    def test_saved_tech_stack_is_used(self):
        assessment = agent.ScoringAssessment(
            justification="AWS is present.",
            rubric_breakdown=agent.RubricBreakdown(
                revenue_fit=100, tech_stack_match=100, segment_fit=100,
            ),
        )
        profile = {"prospect_id": "test-lead", "tech_stack": ["stale"]}
        with patch.object(agent.data_service, "fetch_tech_stack", return_value=["AWS"]), \
                patch.object(agent, "_scoring_llm") as llm:
            llm.invoke.return_value = assessment
            agent.score_prospect.func(profile, self.offering)
        content = llm.invoke.call_args.args[0][1]["content"]
        scored_profile = json.loads(content.split("\n\nProspect profile:\n", 1)[1])
        self.assertEqual(scored_profile["tech_stack"], ["AWS"])
        self.assertEqual(profile["tech_stack"], ["stale"])
