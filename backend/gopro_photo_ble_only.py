#!/usr/bin/env python3
"""
Take a photo on a GoPro using Bluetooth (BLE) only — no WiFi.
The photo is stored on the GoPro's SD card and is NOT downloaded.
"""

import asyncio
import argparse
import logging
from open_gopro import WirelessGoPro


async def take_photo_ble(identifier: str | None = None, log: str = "gopro_ble.log"):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if log:
        fh = logging.FileHandler(log)
        fh.setLevel(logging.DEBUG)
        logging.getLogger().addHandler(fh)

    gopro = WirelessGoPro(
        target=identifier,
        interfaces={WirelessGoPro.Interface.BLE},
    )

    async with gopro:
        print("Connected to GoPro via BLE (Bluetooth only)")

        # Take a photo — the command is sent over BLE
        print("Taking photo...")
        await gopro.ble_command.set_shutter(shutter=True)
        await asyncio.sleep(2)  # Give time for the photo to be captured
        await gopro.ble_command.set_shutter(shutter=False)
        print("Photo taken! It is stored on the GoPro's SD card.")
        print("(No WiFi was used — connect via USB or remove SD card to retrieve it)")


def main():
    parser = argparse.ArgumentParser(description="Take a GoPro photo via Bluetooth only (no WiFi)")
    parser.add_argument("--identifier", help="Last 4 digits of GoPro serial number")
    parser.add_argument("--log", default="gopro_ble.log", help="Log file location")
    args = parser.parse_args()

    asyncio.run(take_photo_ble(identifier=args.identifier, log=args.log))


if __name__ == "__main__":
    main()
