import Link from "next/link";

export default function Home() {
  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center gap-6 p-6">
      <div>
        <h1 className="text-3xl font-semibold">MBAS dataset recorder</h1>
        <p className="mt-2 text-zinc-400">
          Records mock interviews for the Multimodal Behavioural Analysis System dataset: candidate webcam, candidate and
          interviewer microphones, candidate screen, question timestamps and gaze calibration, all on one shared clock.
        </p>
      </div>
      <div className="grid gap-4 sm:grid-cols-2">
        <Link href="/interviewer" className="card transition hover:border-indigo-500">
          <h2 className="font-medium">I am the interviewer</h2>
          <p className="text-sm text-zinc-400">Create a room, ask questions, control recording.</p>
        </Link>
        <Link href="/candidate" className="card transition hover:border-indigo-500">
          <h2 className="font-medium">I am the candidate</h2>
          <p className="text-sm text-zinc-400">Join with the room code from the interviewer.</p>
        </Link>
      </div>
    </main>
  );
}
