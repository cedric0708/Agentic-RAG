"""Kakao Local proxy. Coordinates and credentials are never persisted."""
import asyncio
import os
from typing import Annotated

import httpx
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, ValidationError

router = APIRouter(prefix="/api", tags=["pharmacies"])


class Location(BaseModel):
    id: str
    name: str
    address: str
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class Pharmacy(Location):
    phone: str
    distance_m: float = Field(ge=0)
    place_url: str


async def kakao_search(kind: str, params: dict) -> dict:
    key = os.getenv("KAKAO_REST_API_KEY", "").strip()
    if not key:
        raise HTTPException(503, "약국 검색 서비스가 아직 설정되지 않았습니다.")
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            response = await client.get(
                f"https://dapi.kakao.com/v2/local/search/{kind}.json",
                headers={"Authorization": f"KakaoAK {key}"}, params=params,
            )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data.get("documents"), list) or not isinstance(data.get("meta"), dict):
            raise ValueError("Invalid response")
        return data
    except httpx.TimeoutException:
        raise HTTPException(504, "검색 응답이 지연되고 있습니다. 다시 시도해 주세요.") from None
    except httpx.HTTPStatusError as exc:
        status = 503 if exc.response.status_code in (401, 403, 429) else 502
        raise HTTPException(status, "검색 서비스를 일시적으로 이용할 수 없습니다. 잠시 후 다시 시도해 주세요.") from None
    except (httpx.RequestError, ValueError, TypeError):
        raise HTTPException(502, "검색 서비스에 연결할 수 없습니다. 다시 시도해 주세요.") from None


def location(doc: dict) -> dict:
    return dict(id=doc["id"], name=doc["place_name"],
                address=doc.get("road_address_name") or doc.get("address_name", ""),
                latitude=doc["y"], longitude=doc["x"])


@router.get("/pharmacies")
async def pharmacies(
    lat: Annotated[float, Query(ge=-90, le=90)],
    lng: Annotated[float, Query(ge=-180, le=180)],
    radius: int = 1000,
    page: Annotated[int, Query(ge=1, le=45)] = 1,
):
    if radius not in (500, 1000, 3000):
        raise HTTPException(422, "검색 반경은 500, 1000, 3000m 중 선택해 주세요.")
    data = await kakao_search("category", dict(category_group_code="PM9", x=lng, y=lat,
                                              radius=radius, page=page, size=15, sort="distance"))
    try:
        items = [Pharmacy(**location(d), phone=d.get("phone") or "",
                          distance_m=d["distance"],
                          place_url=f'https://place.map.kakao.com/{d["id"]}').model_dump()
                 for d in data["documents"]]
        is_end = data["meta"]["is_end"]
        if not isinstance(is_end, bool):
            raise ValueError("Invalid pagination")
    except (KeyError, TypeError, ValueError, ValidationError):
        raise HTTPException(502, "검색 결과를 읽을 수 없습니다. 다시 시도해 주세요.") from None
    return {"items": items, "has_next": not is_end and page < 45, "page": page}


@router.get("/locations")
async def locations(query: Annotated[str, Query(min_length=1, max_length=100)]):
    query = query.strip()
    if not query:
        raise HTTPException(422, "지역명 또는 주소를 입력해 주세요.")
    places, addresses = await asyncio.gather(
        kakao_search("keyword", {"query": query, "size": 10}),
        kakao_search("address", {"query": query, "size": 5}),
    )
    try:
        items = [Location(**location(d)).model_dump() for d in places["documents"]]
        for d in addresses["documents"]:
            items.append(Location(id=f'address:{d["x"]},{d["y"]}', name=d["address_name"],
                                  address=d["address_name"], latitude=d["y"], longitude=d["x"]).model_dump())
    except (KeyError, TypeError, ValueError, ValidationError):
        raise HTTPException(502, "지역 검색 결과를 읽을 수 없습니다. 다시 시도해 주세요.") from None
    return {"items": items}
