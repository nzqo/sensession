#!/usr/bin/env python
"""
Small application to handle some things from the command line more conveniently
"""

import shutil
from typing import List
from pathlib import Path

import rich
import typer
from rich.panel import Panel
from rich.table import Table
from rich.console import Console
from typing_extensions import Annotated

from sensession import Database
from sensession.config import APP_CONFIG
from sensession.campaign import (
    Campaign,
    CampaignRunner,
    SerializationMode,
    write_to_disk,
    read_from_disk,
    extract_failed_campaign,
)

app = typer.Typer()


def try_campaign(campaign: Campaign) -> Campaign | None:
    """
    Try to run the campaign; Return a campaign with schedules that failed
    """
    with CampaignRunner(campaign) as runner:
        result = runner.run()
        with Database(f"data/{campaign.name}", append=True) as db:
            runner.store_results(db)

    remaining_camp = extract_failed_campaign(campaign, result)
    return None if remaining_camp.is_empty() else remaining_camp


def overwrite_campaign_file(file: Path, campaigns: List[Campaign]):
    """
    Overwrite campaign file, or remove if it is empty.
    """
    file.unlink()
    for camp in campaigns:
        write_to_disk(camp, file, append=True, mode=SerializationMode.PICKLE)


@app.command()
def rerun_failed(file: Annotated[Path, typer.Argument()], num_tries: int = 1):
    """
    Rerun failed campaigns stored in file at `file`
    """
    rich.traceback.install(show_locals=False)

    if not file.is_file():
        print("File does not exist; nothing to rerun")
        return

    campaigns = read_from_disk(file)
    failed_campaigns = []

    # Run all campaigns stored
    for campaign in campaigns:
        camp = campaign

        # Try for num_tries
        for _ in range(num_tries):
            # If there are no remaining unsuccessful schedules, this campaign is done!
            if (camp := try_campaign(camp)) is None:
                break

        if camp is not None and not camp.is_empty():
            failed_campaigns.append(camp)

    overwrite_campaign_file(file, failed_campaigns)


@app.command()
def inspect_campaign(file: Annotated[Path, typer.Argument()]):
    """
    Inspect the sessions in campaign stored in `file`
    """
    rich.traceback.install(show_locals=False)

    campaigns = read_from_disk(file)

    # Static information
    static_info = (
        f"Campaign file......: {file.resolve()}\n"
        + f"Number of campaigns: {len(campaigns)}\n"
        + f"Number of schedules: {sum(len(c.schedules) for c in campaigns)}"
    )

    # Create a panel for the static information
    static_panel = Panel(static_info, title="Meta info", expand=False)

    console = Console()
    console.print(static_panel)

    for i, camp in enumerate(campaigns):
        if len(camp.schedules) == 0:
            continue

        table = Table(title=f"Campaign {i}")
        table.add_column("Name", style="cyan", justify="center")
        table.add_column("Channel", style="magenta", justify="center")
        table.add_column("Num Events", style="yellow", justify="right")
        for label in camp.schedules[0].extra_labels.keys():
            table.add_column(label, style="blue", justify="center")

        for schedule in camp.schedules:
            table.add_row(
                schedule.name,
                schedule.channel.to_string(),
                str(len(schedule.events)),
                *[str(val) for val in schedule.extra_labels.values()],
            )
        console.print(table)


@app.command()
def clean(data: Annotated[bool, typer.Option(help="Whether to delete data.")] = False):
    """
    Clean temporary data
    """
    shutil.rmtree(".cache")
    if data:
        shutil.rmtree("data")

    Path(APP_CONFIG.logfile).unlink(missing_ok=True)

    raise NotImplementedError()


if __name__ == "__main__":
    app()
