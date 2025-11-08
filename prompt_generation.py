import random
from pathlib import Path
from datetime import datetime
from typing import Optional

class SixLayerPromptGenerator:
    """Final multi-template version + balanced dataset control."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = random.Random(seed)
        self.year = datetime.now().year

        # 1️⃣ Camera configuration layer
        self.camera = {
            "angle": ["top-down", "overhead", "eye-level", "slightly oblique"],
            "height": ["ceiling-mounted at 3.5 m", "wall-mounted at 4.0 m",
                       "pole-mounted at 4.5 m", "steel-beam mounted at 5.0 m"],
            "resolution": [
                "4K clarity (high perceptual fidelity)",
                "1080p HD (balanced clarity)",
                "720p standard quality",
                "grainy CCTV footage (low perceptual fidelity)"
            ],
            "fov": ["wide-angle (180° fisheye)", "medium (~90°)", "narrow (~30°)"],
            "type": ["ATEX-certified fixed camera", "explosion-proof dome camera", "wide-angle CCTV"]
        }

        # 2️⃣ Worker behavior layer
        self.behaviors = [
            ("calling on the phone", ["handheld smartphone to ear", "phone pressed between shoulder and ear", "touchscreen used with gloves"]),
            ("smoking", ["lit cigarette in hand", "visible smoke plume", "ignition ember"]),
            ("sleeping", ["head on forearm", "eyes closed at workstation", "slumped posture"]),
            ("falling", ["losing balance while walking", "slipping near machinery", "falling backward to ground"]),
            ("wearing safety belt", ["securely fastened harness", "belt attached to anchor point"]),
            ("wearing safety helmet", ["helmet properly fitted", "chin strap tightened"])
        ]
        self.ppe = ["PPE compliant", "without safety helmet", "without protective goggles", "reflective vest missing"]

        # 3️⃣ Environment & hazard layer
        self.environments = [
            {"environment": "chemical plant pipeline corridor",
             "equipment": ["valves and gauges", "dense pipelines", "cable trays"],
             "safety": ["warning signs present", "vapor detector nearby"],
             "structure": ["steel frames", "scaffolding", "grating platforms"],
             "hazard": ["oil leakage on the floor", "flames visible near machinery"]},
            {"environment": "tank farm area",
             "equipment": ["pressure vessels", "manifolds", "pumps"],
             "safety": ["warning signs present", "alarm indicator flashing"],
             "structure": ["steel frames", "dense pipelines"],
             "hazard": ["oil leakage near pump base", "flame ignition around valve"]},
            {"environment": "workshop aisle",
             "equipment": ["conveyors", "control panels", "tool cabinets"],
             "safety": ["restricted area markings", "safety placards on wall"],
             "structure": ["steel columns", "cable trays"],
             "hazard": ["localized oil spill on the floor", "small flame under equipment"]},
            {"environment": "control room",
             "equipment": ["control panels", "monitors"],
             "safety": ["warning placards on wall"],
             "structure": ["steel frames"],
             "hazard": ["no visible leakage", "no flame hazard"]}
        ]

        # 4️⃣ Lighting
        self.lighting = [
            ("daytime", "uniform illumination"),
            ("daytime", "dim light"),
            ("daytime", "strong backlight"),
            ("nighttime", "uniform illumination"),
            ("nighttime", "glare")
        ]

        # 5️⃣ Scale
        self.scales = ["close-up", "medium shot", "full-body", "long shot"]

        # 6️⃣ Occlusion
        self.occlusions = ["no occlusion", "partial occlusion", "machinery occlusion", "pipeline blockage", "structural obstruction"]

        # paragraph templates (5 sentence structures)
        self.templates = [
            ("In the {environment}, an {cam_type} positioned {angle} and {height} records {resolution} footage with {fov}. "
             "The camera captures a {scale} view during {time_of_day} under {lighting}, showing a worker who is {behavior} ({detail}){ppe_clause}. "
             "Industrial equipment such as {equip1} and {equip2} can be seen around {structure}, while {safety} is visible. {hazard_text} {occlusion_text}"),
            ("The {environment}, illuminated by {lighting}, is monitored by a {cam_type} {angle} and {height} that records {resolution} footage with {fov}. "
             "Within the frame, a {scale} view reveals a worker {behavior} ({detail}){ppe_clause}. {equip1} and {equip2} extend across {structure}, and {safety} can be observed. {hazard_text} {occlusion_text}"),
            ("A worker {behavior} ({detail}){ppe_clause} is observed in the {environment} during {time_of_day} under {lighting}. "
             "The footage, captured by a {cam_type} {angle} and {height}, provides {resolution} imagery with {fov}. "
             "Around the scene, {equip1} and {equip2} are arranged near {structure}, and {safety} is visible. {hazard_text} {occlusion_text}"),
            ("Under {time_of_day} conditions with {lighting}, a {cam_type} {angle} and {height} records {resolution} footage with {fov} in the {environment}. "
             "The camera captures a {scale} perspective where a worker is {behavior} ({detail}){ppe_clause}. "
             "{equip1} and {equip2} appear around {structure}, and {safety} can be clearly identified. {hazard_text} {occlusion_text}"),
            ("{hazard_prefix} characterizes the monitored scene in the {environment}, captured by a {cam_type} {angle} and {height} under {time_of_day} {lighting}. "
             "The camera produces {resolution} footage with {fov}, presenting a {scale} view where a worker is {behavior} ({detail}){ppe_clause}. "
             "{equip1} and {equip2} fill the background across {structure}, and {safety} is noticeable. {occlusion_text}")
        ]

    # ---------------- helpers ----------------
    def _pick(self, seq): return self.rng.choice(seq)
    def _sample(self, seq, k): return self.rng.sample(seq, k=min(k, len(seq)))

    # ---------------- generate paragraph ----------------
    def generate_prompt(self, fix_behavior=None, fix_hazard=None):
        cam_type = self._pick(self.camera["type"])
        angle = self._pick(self.camera["angle"])
        height = self._pick(self.camera["height"])
        resolution = self._pick(self.camera["resolution"])
        fov = self._pick(self.camera["fov"])

        if fix_behavior:
            candidates = [b for b in self.behaviors if fix_behavior.lower() in b[0]]
            behavior, details = self._pick(candidates) if candidates else self._pick(self.behaviors)
        else:
            behavior, details = self._pick(self.behaviors)
        detail = self._pick(details)
        ppe_state = self._pick(self.ppe)
        ppe_clause = "" if ppe_state == "PPE compliant" else f", {ppe_state}"

        env = self._pick(self.environments)
        environment = env["environment"]
        equip1, equip2 = self._sample(env["equipment"], 2)
        safety = self._pick(env["safety"])
        structure = self._pick(env["structure"])
        hazard = fix_hazard if fix_hazard else self._pick(env["hazard"])

        time_of_day, lighting = self._pick(self.lighting)
        scale = self._pick(self.scales)
        occlusion = self._pick(self.occlusions)

        if hazard.lower().startswith("no "):
            hazard_text = ""
            hazard_prefix = "No specific hazard"
        else:
            hazard_text = f"In addition, {hazard} is present within the monitored area."
            hazard_prefix = hazard.capitalize()
        if occlusion == "no occlusion":
            occlusion_text = "The scene remains unobstructed, providing a clear and stable surveillance perspective."
        else:
            occlusion_text = f"The footage contains {occlusion}, reflecting realistic surveillance conditions in complex environments."

        tmpl = self._pick(self.templates)
        return " ".join(tmpl.format(environment=environment, cam_type=cam_type, angle=angle, height=height,
                                    resolution=resolution, fov=fov, scale=scale, time_of_day=time_of_day, lighting=lighting,
                                    behavior=behavior, detail=detail, ppe_clause=ppe_clause,
                                    equip1=equip1, equip2=equip2, structure=structure, safety=safety,
                                    hazard_text=hazard_text, occlusion_text=occlusion_text, hazard_prefix=hazard_prefix).split())

    # ---------------- balanced dataset generation ----------------
    def generate_balanced_dataset(self, category="behavior", each_count=500, save_path="prompts_balanced"):
        path = Path(save_path).resolve()
        path.mkdir(parents=True, exist_ok=True)
        out = path / f"balanced_{category}_{each_count}.txt"

        uniq = set()
        with open(out, "w", encoding="utf-8") as f:
            if category == "behavior":
                for behavior_name, _ in self.behaviors:
                    for _ in range(each_count):
                        p = self.generate_prompt(fix_behavior=behavior_name)
                        uniq.add(p)
                        f.write(p + "\n")
                    print(f"✅ Generated {each_count} prompts for behavior: {behavior_name}")

            elif category == "hazard":
                all_hazards = ["oil leakage", "flames"]
                for hz in all_hazards:
                    for _ in range(each_count):
                        p = self.generate_prompt(fix_hazard=hz)
                        uniq.add(p)
                        f.write(p + "\n")
                    print(f"✅ Generated {each_count} prompts for hazard: {hz}")
            else:
                print("⚠️ category must be 'behavior' or 'hazard'")

        print(f"\n✅ Successfully generated {len(uniq)} total prompts")
        print(f"📁 Saved to: {out}")


if __name__ == "__main__":
    gen = SixLayerPromptGenerator(seed=7)
    gen.generate_balanced_dataset(category="behavior", each_count=500, save_path="output")

    # example
    # gen.generate_balanced_dataset(category="hazard", each_count=500, save_path="C:/Users/52844/Desktop/ChemSafe_Prompts")
