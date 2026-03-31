import click
from pathlib import Path
from src.pipeline import Pipeline, RunConfig, configs, load_run_config, preprocess_configs
from src.pdfcrawl_dataset import prepare_pdfcrawl_dataset

@click.group()
def cli():
    """金融研报问答 RAG 系统命令行工具"""
    pass

@cli.command()
def download_models():
    """下载 Docling 所需模型"""
    click.echo("正在下载 Docling 模型...")
    Pipeline.download_docling_models()


@cli.command("prepare-pdfcrawl-dataset")
@click.option('--pdfcrawl-root', type=click.Path(exists=True, file_okay=False, path_type=Path), required=True, help='PDFCrawl output 根目录')
@click.option('--dataset-dir', type=click.Path(file_okay=False, path_type=Path), required=True, help='输出数据集目录')
@click.option('--link-mode', type=click.Choice(['symlink', 'copy']), default='symlink', show_default=True, help='PDF 落盘方式')
@click.option('--currency', default='CNY', show_default=True, help='写入 manifest 的默认币种')
@click.option('--language', default='zh', show_default=True, help='写入 manifest 的文档语言')
@click.option('--write-questions-stub/--no-write-questions-stub', default=True, show_default=True, help='是否生成空 questions.json')
def prepare_pdfcrawl_dataset_command(pdfcrawl_root, dataset_dir, link_mode, currency, language, write_questions_stub):
    """把 PDFCrawl 输出整理成 FinaRAG 可直接消费的数据集目录"""
    summary = prepare_pdfcrawl_dataset(
        pdfcrawl_root=pdfcrawl_root,
        dataset_dir=dataset_dir,
        link_mode=link_mode,
        currency=currency,
        language=language,
        write_questions_stub=write_questions_stub,
    )
    click.echo(
        f"已生成数据集: {summary.dataset_dir}\n"
        f"- manifests: {len(summary.manifest_paths)}\n"
        f"- documents: {summary.documents_written}\n"
        f"- skipped duplicates: {summary.skipped_rows}\n"
        f"- link mode: {summary.link_mode}"
    )

@cli.command()
@click.option('--parallel/--sequential', default=True, help='并行或顺序解析模式')
@click.option('--chunk-size', default=2, help='每个 worker 处理的 PDF 数量')
@click.option('--max-workers', default=10, help='并行 worker 进程数')
@click.option('--cuda-devices', default=None, help='并行解析时绑定的 CUDA 设备列表，例如 0,1')
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help='YAML 配置文件路径')
def parse_pdfs(parallel, chunk_size, max_workers, cuda_devices, config_path):
    """解析 PDF 研报（支持并行处理）"""
    root_path = Path.cwd()
    run_config = load_run_config(config_path) if config_path else RunConfig()
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(
        f"解析 PDF (config={config_path or 'default'}, parallel={parallel}, "
        f"chunk_size={chunk_size}, max_workers={max_workers}, cuda_devices={cuda_devices or 'default'})"
    )
    pipeline.parse_pdf_reports(
        parallel=parallel,
        chunk_size=chunk_size,
        max_workers=max_workers,
        cuda_devices=cuda_devices,
    )

@cli.command()
@click.option('--max-workers', default=10, help='表格序列化并行 worker 数')
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help='YAML 配置文件路径')
def serialize_tables(max_workers, config_path):
    """对解析后的研报中的表格进行序列化处理"""
    root_path = Path.cwd()
    run_config = load_run_config(config_path) if config_path else RunConfig()
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"表格序列化处理 (config={config_path or 'default'}, max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='选择配置预设')
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help='YAML 配置文件路径')
def process_reports(config, config_path):
    """执行文档处理流水线（合并、切块、向量化）"""
    root_path = Path.cwd()
    run_config = load_run_config(config_path) if config_path else preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"处理研报文档 (config={config_path or config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['qwen_base', 'qwen_vector_rerank', 'qwen_rerank', 'qwen_sparse_rerank', 'qwen_ser_vector_rerank', 'qwen_ser_rerank', 'qwen_ser_sparse_rerank']), default='qwen_base', help='选择配置预设')
@click.option('--config-path', type=click.Path(exists=True, dir_okay=False, path_type=Path), default=None, help='YAML 配置文件路径')
def process_questions(config, config_path):
    """基于指定配置处理问答"""
    root_path = Path.cwd()
    run_config = load_run_config(config_path) if config_path else configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"处理问答 (config={config_path or config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()
    
