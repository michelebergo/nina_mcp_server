import asyncio
import aiohttp
import json

async def test_weather_endpoints():
    base_url = "http://100.121.20.60:1888/v2/api"
    
    async with aiohttp.ClientSession() as session:
        print("=" * 80)
        print("TESTING WEATHER ENDPOINTS")
        print("=" * 80)
        
        # Test 1: Weather Info
        print("\n1. GET equipment/weather/info")
        print("-" * 80)
        try:
            async with session.get(f"{base_url}/equipment/weather/info") as response:
                print(f"Status: {response.status}")
                data = await response.json()
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 2: List Weather Devices
        print("\n\n2. GET equipment/weather/list-devices")
        print("-" * 80)
        try:
            async with session.get(f"{base_url}/equipment/weather/list-devices") as response:
                print(f"Status: {response.status}")
                data = await response.json()
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        # Test 3: Rescan Weather Devices
        print("\n\n3. GET equipment/weather/rescan")
        print("-" * 80)
        try:
            async with session.get(f"{base_url}/equipment/weather/rescan") as response:
                print(f"Status: {response.status}")
                data = await response.json()
                print(json.dumps(data, indent=2))
        except Exception as e:
            print(f"Error: {e}")
        
        print("\n" + "=" * 80)
        print("TESTING COMPLETE")
        print("=" * 80)

if __name__ == "__main__":
    asyncio.run(test_weather_endpoints())
