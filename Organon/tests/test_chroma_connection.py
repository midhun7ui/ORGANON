
import asyncio
import chromadb


async def main():
    # Connect to the ChromaDB server running in Docker.
    client = await chromadb.AsyncHttpClient(
        host="localhost",
        port=8000,
    )

    # Check whether the server is responding.
    heartbeat = await client.heartbeat()

    print("Connected to ChromaDB!")
    print(f"Heartbeat: {heartbeat}")


if __name__ == "__main__":
    asyncio.run(main())
