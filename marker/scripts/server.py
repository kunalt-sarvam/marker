import traceback
import uuid

import click
import os

from pydantic import BaseModel, Field
from starlette.responses import HTMLResponse

from marker.config.parser import ConfigParser

from contextlib import asynccontextmanager
from typing import Optional, Annotated

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.staticfiles import StaticFiles
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.renderers.json import JSONOutput
from marker.settings import settings
from temporalio.client import Client
from marker.scripts.temporal_workflow import PDFConversionWorkflow, PDFConversionParams

app_data = {}


UPLOAD_DIRECTORY = "./uploads"
CONVERSION_RESULTS_DIR = settings.OUTPUT_DIR
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
os.makedirs(CONVERSION_RESULTS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_data["models"] = create_model_dict()

    yield

    if "models" in app_data:
        del app_data["models"]


app = FastAPI(lifespan=lifespan)

app.mount("/results", StaticFiles(directory=CONVERSION_RESULTS_DIR), name="results")


class CommonParams(BaseModel):
    filepath: Annotated[
        Optional[str], Field(description="The path to the PDF file to convert.")
    ]
    page_range: Annotated[
        Optional[str],
        Field(
            description="Page range to convert, specify comma separated page numbers or ranges.  Example: 0,5-10,20"
        ),
    ] = None


async def _convert_pdf(params: CommonParams):
    options = params.model_dump()
    options["output_format"] = "json"
    options["force_ocr"] = True

    config_parser = ConfigParser(options)
    config_dict = config_parser.generate_config_dict()
    config_dict["pdftext_workers"] = 1
    converter_cls = PdfConverter
    converter = converter_cls(
        config=config_dict,
        artifact_dict=app_data["models"],
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    rendered = converter(params.filepath)

    assert isinstance(rendered, JSONOutput)

    return rendered


@app.post("/marker/upload")
async def convert_pdf_upload(
    page_range: Optional[str] = Form(default=None),
    file: UploadFile = File(
        ..., description="The PDF file to convert.", media_type="application/pdf"
    ),
):
    filename_base = os.path.splitext(file.filename)[0]
    file_dir = os.path.join(CONVERSION_RESULTS_DIR, filename_base)
    os.makedirs(file_dir, exist_ok=True)

    upload_path = os.path.join(UPLOAD_DIRECTORY, file.filename)
    with open(upload_path, "wb+") as upload_file:
        file_contents = await file.read()
        upload_file.write(file_contents)

    temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
    client = await Client.connect(temporal_host)
    workflow_id = f"pdf-conversion-{uuid.uuid4()}"

    workflow_params = PDFConversionParams(
        filepath=upload_path,
        page_range=page_range,
        filename_base=filename_base,
    )

    try:
        await client.start_workflow(
            PDFConversionWorkflow,
            workflow_params,
            id=workflow_id,
            task_queue="pdf-conversion-queue",
        )

        return {
            "success": True,
            "workflow_id": workflow_id,
            "status": "processing",
            "message": f"PDF conversion started. Check status with GET /marker/status/{workflow_id}",
        }
    except Exception as e:
        traceback.print_exc()
        os.remove(upload_path)
        return {
            "success": False,
            "error": str(e),
        }


@app.get("/marker/status/{workflow_id}")
async def get_conversion_status(workflow_id: str):
    try:
        temporal_host = os.getenv("TEMPORAL_HOST", "localhost:7233")
        client = await Client.connect(temporal_host)
        handle = client.get_workflow_handle(workflow_id)

        description = await handle.describe()

        if description.status is None:
            return {
                "success": True,
                "workflow_id": workflow_id,
                "status": "running",
                "message": "Workflow is still processing",
            }
        elif description.status.name == "COMPLETED":
            result = await handle.result()
            return {**result, "workflow_id": workflow_id, "status": "completed"}
        elif description.status.name == "FAILED":
            return {
                "success": False,
                "workflow_id": workflow_id,
                "status": "failed",
                "error": "Workflow failed",
            }
        else:
            return {
                "success": False,
                "workflow_id": workflow_id,
                "status": description.status.name.lower(),
                "message": f"Workflow status: {description.status.name}",
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "status": "error",
        }


@click.command()
@click.option("--port", type=int, default=8000, help="Port to run the server on")
@click.option("--host", type=str, default="127.0.0.1", help="Host to run the server on")
@click.option("--reload", is_flag=True, default=False, help="Enable auto-reload")
def server_cli(port: int, host: str, reload: bool):
    import uvicorn

    if reload:
        uvicorn.run("marker.scripts.server:app", host=host, port=port, reload=True)
    else:
        uvicorn.run(app, host=host, port=port)
