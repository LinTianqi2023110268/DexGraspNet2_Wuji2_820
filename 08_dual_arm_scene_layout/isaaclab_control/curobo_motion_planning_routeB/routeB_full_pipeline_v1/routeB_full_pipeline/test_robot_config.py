import unittest

from routeB_full_pipeline.robot_config import ATTACHED_LINK, with_attachment_link


class RobotConfigTest(unittest.TestCase):
    def test_additive_attachment_link(self):
        src = {
            "kinematics": {
                "collision_link_names": ["arm_r_link_tf"],
                "extra_links": {},
                "extra_collision_spheres": None,
            }
        }
        out = with_attachment_link(src, sphere_slots=17)
        kin = out["kinematics"]
        self.assertIn(ATTACHED_LINK, kin["extra_links"])
        self.assertIn(ATTACHED_LINK, kin["collision_link_names"])
        self.assertEqual(kin["extra_collision_spheres"][ATTACHED_LINK], 17)
        self.assertNotIn(ATTACHED_LINK, src["kinematics"]["extra_links"])


if __name__ == "__main__":
    unittest.main()
