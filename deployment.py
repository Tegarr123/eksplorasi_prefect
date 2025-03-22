from prefect import flow


if __name__ == '__main__':
    flow.from_source(
        source="https://github.com/Tegarr123/eksplorasi_prefect.git",
        entrypoint="main.py:mnist_workflow_ml"
    ).deploy(
        name = 'deployment_1',
        work_pool_name='DC-work-pool',
        tags=['dev'],
        job_variables={'pip_packages':['matplotlib','tqdm','torchmetrics','torch','torchvision', 'prefect']}
    )