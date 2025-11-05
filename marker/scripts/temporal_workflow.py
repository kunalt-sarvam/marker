import asyncio
from datetime import timedelta
from temporalio import workflow, activity
from temporalio.client import Client
from temporalio.worker import Worker
from pydantic import BaseModel
import os


class PDFConversionParams(BaseModel):
    filepath: str
    page_range: str | None = None
    filename_base: str


@workflow.defn
class PDFConversionWorkflow:
    @workflow.run
    async def run(self, params: PDFConversionParams) -> dict:
        # This will be the activity
        return await workflow.execute_activity(
            convert_pdf_activity,
            params,
            start_to_close_timeout=timedelta(minutes=30),
        )


@activity.defn
async def convert_pdf_activity(params: PDFConversionParams) -> dict:
    # Import heavy libraries inside the activity to avoid workflow sandbox issues
    from marker.converters.pdf import PdfConverter
    from marker.config.parser import ConfigParser
    from marker.models import create_model_dict
    from marker.renderers.json import JSONOutput
    from marker.settings import settings
    from json_to_html import MarkerJSONToHTML

    options = {
        "filepath": params.filepath,
        "page_range": params.page_range,
        "output_format": "json",
        "force_ocr": True,
    }

    config_parser = ConfigParser(options)
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1

    # Load models (this might need to be cached or loaded once)
    models = create_model_dict()

    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=models,
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(params.filepath)

    assert isinstance(rendered, JSONOutput)

    # Save results
    CONVERSION_RESULTS_DIR = settings.OUTPUT_DIR
    file_dir = os.path.join(CONVERSION_RESULTS_DIR, params.filename_base)
    os.makedirs(file_dir, exist_ok=True)

    json_path = os.path.join(file_dir, f"{params.filename_base}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(rendered.model_dump_json(indent=2))

    base_url = f"/results/{params.filename_base}"
    converter_html = MarkerJSONToHTML(base_url=base_url)
    html_path = os.path.join(file_dir, "index.html")
    converter_html.convert(json_path, html_path)

    return {
        "result_url": f"/results/{params.filename_base}/index.html",
    }


async def start_worker():
    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)

    worker = Worker(
        client,
        task_queue="pdf-conversion-queue",
        workflows=[PDFConversionWorkflow],
        activities=[convert_pdf_activity],
    )

    await worker.run()


if __name__ == "__main__":
    asyncio.run(start_worker())
