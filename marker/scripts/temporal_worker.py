import asyncio
import os
from temporalio.client import Client
from temporalio.worker import Worker
from marker.scripts.temporal_workflow import PDFConversionWorkflow, convert_pdf_activity


async def main():
    client = await Client.connect(os.getenv("TEMPORAL_HOST", "localhost:7233"))

    worker = Worker(
        client,
        task_queue="pdf-conversion-queue",
        workflows=[PDFConversionWorkflow],
        activities=[convert_pdf_activity],
    )

    print("Starting Temporal worker for PDF conversion...")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
