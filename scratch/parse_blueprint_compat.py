import re
import sys
from pathlib import Path

source_file = Path("balatro_source/game.lua")
with open(source_file, "r", encoding="utf-8") as f:
    content = f.read()

# Match lines inside G.P_CENTERS definition for jokers
# Format: j_name = { order = ..., name = "...", ... blueprint_compat = true/false ... }
pattern = re.compile(r"^\s*(j_\w+)\s*=\s*\{(.+?)\},?\s*$", re.MULTILINE)

jokers_compat = {}
for match in pattern.finditer(content):
    var_name = match.group(1)
    body = match.group(2)
    if "set = 'Joker'" in body or 'set = "Joker"' in body:
        # Extract name
        name_match = re.search(r'name\s*=\s*["\']([^"\']+)["\']', body)
        bp_match = re.search(r'blueprint_compat\s*=\s*(true|false)', body)
        if name_match:
            joker_name = name_match.group(1)
            # Default to False if blueprint_compat is missing, but check if bp_match exists
            is_compat = (bp_match.group(1) == "true") if bp_match else False
            jokers_compat[joker_name] = (is_compat, bp_match is not None)

print(f"Total Jokers Found: {len(jokers_compat)}")
compat_jokers = [name for name, (comp, has_field) in jokers_compat.items() if comp]
incompat_jokers = [name for name, (comp, has_field) in jokers_compat.items() if not comp]

print(f"Compatible: {len(compat_jokers)}")
print(f"Incompatible: {len(incompat_jokers)}")

print("\n--- INCOMPATIBLE JOKERS ---")
for name in sorted(incompat_jokers):
    print(f"'{name}',")

print("\n--- COMPATIBLE JOKERS ---")
for name in sorted(compat_jokers):
    print(f"'{name}',")
