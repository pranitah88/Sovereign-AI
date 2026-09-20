import { RobotHero } from "@/components/ui/robot-hero";

const settings = {
  color: "#c4c4c4",
  scale: 1,
  pantallaColor: "#00ffc6",
  pantallaBrillo: 1.2,
  blinkCycle: 3.0,
  metalness: 0.0,
};

export default function Demo(props: Partial<typeof settings>) {
  const s = { ...settings, ...props };
  return (
    <div className="h-screen w-screen">
      <RobotHero {...s} />
    </div>
  );
}
