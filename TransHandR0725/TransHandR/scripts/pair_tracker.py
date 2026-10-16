#!/usr/bin/env python3
"""
VIVE Tracker 3.0 Pairing Tool for Wireless Dongle (Linux, no SteamVR)
Uses pyusb to send RF init + pairing commands directly.

Usage:
  1. Plug in VIVE Wireless Dongle
  2. Power on VIVE Tracker 3.0, HOLD power button until LED fast-blinks BLUE
  3. Run: sudo python pair_tracker.py

Prerequisites:
  pip install pyusb
"""

import subprocess, usb.core, usb.util, time, os, sys

DONGLE_VID = 0x0BB4
DONGLE_PID = 0x0350

def sudo_cmd(cmd):
    return subprocess.run(["sudo", "-S", "bash", "-c", cmd],
                         input="098765\n", text=True, capture_output=True)

def main():
    # 1. Find dongle
    dev = usb.core.find(idVendor=DONGLE_VID, idProduct=DONGLE_PID)
    if dev is None:
        print("ERROR: VIVE Wireless Dongle not found!")
        sys.exit(1)

    bus = dev.bus
    addr = dev.address
    print(f"Found dongle: bus={bus}, addr={addr}")

    # 2. Unbind kernel driver + set permissions
    usb_path = f"/dev/bus/usb/{bus:03d}/{addr:03d}"
    print("Unbinding kernel driver...")
    sudo_cmd(f"echo -n '{bus}-1:1.0' > /sys/bus/usb/drivers/usbhid/unbind 2>&1")
    time.sleep(0.3)
    sudo_cmd(f"chmod 666 {usb_path}")

    # 3. Claim device via pyusb
    try:
        dev.set_configuration()
        usb.util.claim_interface(dev, 0)
        print("Device claimed successfully")
    except Exception as e:
        print(f"ERROR claiming device: {e}")
        sys.exit(1)

    # 4. Send VIVE commands
    def send_cmd(payload, desc=""):
        try:
            dev.ctrl_transfer(0x21, 0x09, 0x0300, 0, bytes(payload), timeout=1000)
            print(f"  OK: {desc}")
            return True
        except usb.core.USBError as e:
            print(f"  FAIL: {desc} - {e}")
            return False

    print("\n=== RF Protocol Init ===")
    send_cmd([0xFF, 0x87, 0x06, 0x01, 0x00, 0x00, 0x02, 0x00, 0x00], "RF raw mode 0")
    time.sleep(0.15)

    magic = [0xFF, 0x87, 0x03, 0x00, 0x01, 0x00]
    send_cmd(magic, "Protocol super magic")
    time.sleep(0.15)
    send_cmd(magic, "Protocol switch")
    time.sleep(0.15)

    print("\n=== PAIRING ===")
    print(">> Make sure tracker LED is FAST-BLINKING BLUE <<")
    pair = [0xFF, 0xAD, 0x03, 0x01, 0x10, 0x27]
    send_cmd(pair, "PAIRING COMMAND")

    # 5. Monitor data
    print("\n=== Monitoring data (30s, Ctrl+C to stop) ===")
    print("Waiting for tracker data... LED should turn SOLID GREEN if paired")
    start = time.time()
    pkt = 0
    types = {}
    try:
        while time.time() - start < 30:
            try:
                data = dev.read(0x81, 64, timeout=200)
                if data and len(data):
                    pkt += 1
                    t = time.time() - start
                    rid = data[0]
                    names = {0x1E: "HEARTBEAT", 0x20: "IMU", 0x23: "LIGHTHOUSE",
                             0x24: "LIGHTCAP", 0xFF: "CMD_RESP"}
                    rn = names.get(rid, f"UNKNOWN(0x{rid:02X})")
                    types[rn] = types.get(rn, 0) + 1
                    print(f"  [{t:5.1f}s] 0x{rid:02X} {rn:12s} | {' '.join(f'{b:02x}' for b in data[:16])}")
            except usb.core.USBTimeoutError:
                pass
            except KeyboardInterrupt:
                break
    except KeyboardInterrupt:
        pass

    print(f"\n=== Summary ===")
    print(f"Total packets: {pkt}")
    for tname, count in sorted(types.items()):
        print(f"  {tname}: {count}")

    # 6. Cleanup
    print("\nRe-attaching kernel driver...")
    usb.util.dispose_resources(dev)
    sudo_cmd(f"echo -n '{bus}-1:1.0' > /sys/bus/usb/drivers/usbhid/bind")
    print("Done")

if __name__ == "__main__":
    main()
