import asyncio
import aiohttp
import aiofiles
import json
import trafilatura
from tqdm import tqdm

INPUT = "data/news/raw/20140101_urls.jsonl"
OUTPUT = "data/news/raw/20140101_articles.jsonl"

MAX_CONCURRENT = 20


async def fetch(session, url):

    try:
        async with session.get(url, timeout=20) as r:

            if r.status != 200:
                return None

            html = await r.text(errors="ignore")

            text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False
            )

            if not text:
                return None

            return text

    except:
        return None


async def worker(session, row, outfile, sem):

    async with sem:

        urls = row["source"].split("<UDIV>")

        for url in urls:

            url = url.split("&c=")[0].strip()

            text = await fetch(session, url)

            if text:

                obj = {
                    "gkg_id": row["gkg_id"],
                    "date": row["date"],
                    "url": url,
                    "text": text
                }

                await outfile.write(
                    json.dumps(obj, ensure_ascii=False) + "\n"
                )


async def main():

    sem = asyncio.Semaphore(MAX_CONCURRENT)

    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT)

    async with aiohttp.ClientSession(connector=connector) as session:

        async with aiofiles.open(
            OUTPUT,
            "w",
            encoding="utf-8"
        ) as outfile:

            tasks = []

            with open(INPUT, encoding="utf-8") as f:

                for line in f:

                    row = json.loads(line)

                    tasks.append(
                        worker(
                            session,
                            row,
                            outfile,
                            sem
                        )
                    )

            for t in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                await t


asyncio.run(main())