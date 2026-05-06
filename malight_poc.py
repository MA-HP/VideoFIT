#!/usr/bin/env python3
import socket


class SimpleLightingControl:

    def __init__(self, ip_address="169.254.5.100", port=62077):
        self.ip_address = ip_address
        self.port = port
        self.CHANNEL_PREFIX = "CH"

    def send_command(self, command: str) -> str:
        """Send command to device"""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            sock.connect((self.ip_address, self.port))

            # Ensure command has newline
            if not command.endswith('\n'):
                command += '\n'

            print(f"Raw bytes: {repr(command.encode('ascii'))}")  # DEBUG
            sock.sendall(command.encode('ascii'))

            response = sock.recv(8192).decode('ascii', errors='ignore')
            sock.close()
            return response
        except Exception as e:
            print(f"Error: {e}")
            return None

    def get_intensity_command(self, channel: int, intensity: float) -> str:
        """Generate intensity command"""
        intensity_str = f"{intensity:.2f}".rstrip('0').rstrip('.')
        cmd = f"{self.CHANNEL_PREFIX}{channel},{intensity_str}\n"  # ← ENSURE \n
        return cmd

    def get_on_command(self, channel: int) -> str:
        """Generate ON command"""
        return f"{self.CHANNEL_PREFIX}{channel},ON\n"

    def get_off_command(self, channel: int) -> str:
        """Generate OFF command"""
        return f"{self.CHANNEL_PREFIX}{channel},OFF\n"

    def step_1_choose_channel(self) -> int:
        print("\n" + "=" * 60)
        print("STEP 1: CHOOSE CHANNEL")
        print("=" * 60)
        print("\nChannels:")
        print("  1 = CH1 (EPI1)")
        print("  2 = CH2 (COAX)")
        print("  3 = CH3 (EPI2)")
        print("  4 = CH4 (DIA)")

        while True:
            channel = input("\nEnter channel (1-4): ").strip()
            if channel in ['1', '2', '3', '4']:
                return int(channel)
            print("Invalid. Enter 1-4")

    def step_2_set_value(self, channel: int) -> str:
        print("\n" + "=" * 60)
        print(f"STEP 2: SET VALUE FOR CH{channel}")
        print("=" * 60)
        print("\nOptions:")
        print("  ON      = Turn on (max intensity)")
        print("  OFF     = Turn off")
        print("  0-100   = Set intensity (0.0 to 100.0)")

        while True:
            value = input(f"\nEnter value (ON/OFF/0-100): ").strip().upper()

            if value == "ON":
                return self.get_on_command(channel)
            elif value == "OFF":
                return self.get_off_command(channel)
            else:
                try:
                    intensity = float(value)
                    if 0 <= intensity <= 100:
                        return self.get_intensity_command(channel, intensity)
                    else:
                        print("Must be 0-100")
                except ValueError:
                    print("Invalid. Use ON, OFF, or 0-100")

    def run(self):
        print("\n" + "=" * 60)
        print("MA LIGHTING CONTROLLER - SIMPLE CONTROL")
        print("=" * 60)
        print(f"Device: {self.ip_address}:{self.port}\n")

        while True:
            try:
                channel = self.step_1_choose_channel()
                command = self.step_2_set_value(channel)

                print("\n" + "-" * 60)
                print(f"Sending: {repr(command)}")  # Show with \n visible
                response = self.send_command(command)

                if response:
                    print("✅ SUCCESS!")
                    print(f"Device says: {response[:100]}")
                else:
                    print("❌ FAILED")

                print("-" * 60)
                again = input("\nContinue? (y/n): ").strip().lower()
                if again != 'y':
                    print("Goodbye!")
                    break

            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break


if __name__ == "__main__":
    controller = SimpleLightingControl()
    controller.run()