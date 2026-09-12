# Q4 v2: 保留已验证的定向定位和覆盖兜底, 叠加联合选路及 22 点证书.
from __future__ import annotations

from robot.agent_fast_v2 import Q3FastAgentV2
from robot.client import ApiClient
from robot.fast_geometry import Arr
from robot.fast_geometry_v2 import compact_q4_sites, direction_certificate
from robot.q4_agent_fast import Q4FastAgent, q4_ring_sites


class Q4FastAgentV2(Q4FastAgent, Q3FastAgentV2):
  def __init__(self, client: ApiClient, err_deg: float = 1.01,
               opportunistic: bool = True, inline: bool = True,
               scan: str = "compact") -> None:
    if scan not in {"compact", "rings", "lattice"}:
      raise ValueError("Q4 scan must be compact, rings or lattice")
    self.use_compact = scan == "compact"
    self.certificate_cells = 0
    self.compact_certified = False
    super().__init__(client, err_deg, opportunistic, inline,
                     "rings" if self.use_compact else scan)

  def _scan_sites(self) -> Arr:
    if not self.use_compact:
      return super()._scan_sites()
    sites = compact_q4_sites()
    ok, self.certificate_cells = direction_certificate(sites)
    self.compact_certified = ok
    # 认证在 /enter 之前执行, 精度或计算预算不足就使用原 25 点解析证书.
    return sites if ok else q4_ring_sites()
