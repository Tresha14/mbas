"use client";

import { Header, PageTitle } from "@/components/shell";
import { ModelList } from "@/components/models";

export default function ModelsPage() {
  return (
    <>
      <Header />
      <main className="mx-auto max-w-6xl space-y-8 px-6 py-10">
        <PageTitle eyebrow="Under the hood" title="Models" subtitle="Every model in the pipeline, what it does and how it scored on held-out data." />
        <ModelList />
      </main>
    </>
  );
}
