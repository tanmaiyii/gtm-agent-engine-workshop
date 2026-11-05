import unittest
from copy import deepcopy
from unittest.mock import patch

from gtm_agent import data_service


class UpdateProspectInfoTests(unittest.TestCase):
    def setUp(self):
        self.records = deepcopy(data_service.PROSPECTS)
        self.records_patch = patch.object(data_service, "PROSPECTS", self.records)
        self.records_patch.start()
        self.addCleanup(self.records_patch.stop)
        self.profiles_patch = patch.object(data_service, "_PROFILES", {})
        self.profiles_patch.start()
        self.addCleanup(self.profiles_patch.stop)

    def test_updates_are_visible_to_source_and_cached_profile_reads(self):
        for prospect_id, technology in (
            ("LEAD-39002", "Terraform"),
            ("LEAD-71001", "Kafka"),
        ):
            with self.subTest(prospect_id=prospect_id):
                profile = deepcopy(data_service.get_prospect_record(prospect_id))
                self.assertNotIn(technology, profile["tech_stack"])
                data_service.save_profile_to_db(prospect_id, profile)

                result = data_service.update_prospect_info(prospect_id, technology)

                self.assertTrue(result["updated"])
                self.assertTrue(result["found"])
                self.assertIn(technology, result["tech_stack"])
                self.assertEqual(
                    data_service.fetch_tech_stack(prospect_id), result["tech_stack"]
                )
                self.assertEqual(
                    data_service.get_prospect_record(prospect_id)["tech_stack"],
                    result["tech_stack"],
                )
                cached = data_service.get_profile_from_db(prospect_id)["prospect_profile"]
                self.assertEqual(cached["tech_stack"], result["tech_stack"])

    def test_update_without_cached_profile_and_duplicate_addition(self):
        prospect_id = "LEAD-39002"
        original = list(data_service.fetch_tech_stack(prospect_id))

        data_service.update_prospect_info(prospect_id, "Terraform")
        result = data_service.update_prospect_info(prospect_id, "Terraform")

        self.assertEqual(result["tech_stack"], original + ["Terraform"])
        self.assertEqual(data_service.fetch_tech_stack(prospect_id), result["tech_stack"])
        self.assertIsNone(
            data_service.get_profile_from_db(prospect_id)["prospect_profile"]
        )

    def test_unknown_prospect_is_not_updated(self):
        self.assertEqual(
            data_service.update_prospect_info("unknown", "Terraform"),
            {"updated": False, "found": False},
        )
        self.assertIsNone(data_service.get_prospect_record("unknown"))


if __name__ == "__main__":
    unittest.main()
