import click
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

@click.group()
def cli():
    """金融研报问答 RAG 系统命令行工具"""
    pass

@cli.command()
def download_models():
    """下载 Docling 所需模型"""
    click.echo("正在下载 Docling 模型...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='并行或顺序解析模式')
@click.option('--chunk-size', default=2, help='每个 worker 处理的 PDF 数量')
@click.option('--max-workers', default=10, help='并行 worker 进程数')
def parse_pdfs(parallel, chunk_size, max_workers):
    """解析 PDF 研报（支持并行处理）"""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"解析 PDF (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='表格序列化并行 worker 数')
def serialize_tables(max_workers):
    """对解析后的研报中的表格进行序列化处理"""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)
    
    click.echo(f"表格序列化处理 (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='选择配置预设')
def process_reports(config):
    """执行文档处理流水线（合并、切块、向量化）"""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"处理研报文档 (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['qwen_base', 'qwen_vector_rerank', 'qwen_rerank', 'qwen_sparse_rerank', 'qwen_ser_vector_rerank', 'qwen_ser_rerank', 'qwen_ser_sparse_rerank']), default='qwen_base', help='选择配置预设')
def process_questions(config):
    """基于指定配置处理问答"""
    root_path = Path.cwd()
    run_config = configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"处理问答 (config={config})...")
    pipeline.process_questions()

if __name__ == '__main__':
    cli()
    
