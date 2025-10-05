export type ContractSummary = {
  contractId: string;
  symbol: string;
  description: string;
  tickSize: number;
  productCode?: string;
};

export type BarPoint = {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
};

export type OrderTicketPayload = {
  accountId: string;
  contractId: string;
  quantity: number;
  side: "BUY" | "SELL";
  orderType: "MARKET" | "LIMIT";
  price?: number;
};

export type AccountSummary = {
  accountId: string;
  name: string;
};

const DEFAULT_BASE = "";

export class GatewayClient {
  constructor(private readonly apiKey: string, private readonly baseUrl = DEFAULT_BASE) {}

  private async post<T>(path: string, body: Record<string, unknown>): Promise<T> {
    const response = await fetch(`${this.baseUrl}${path}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": this.apiKey,
      },
      body: JSON.stringify(body),
    });
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(detail || response.statusText);
    }
    return (await response.json()) as T;
  }

  searchContracts(query: string): Promise<{ items: ContractSummary[] }> {
    return this.post("/gateway/contracts/search", { query });
  }

  searchAccounts(): Promise<{ items: AccountSummary[] }> {
    return this.post("/gateway/accounts/search", {});
  }

  loadBars(contractId: string, timeframe: string, limit = 100): Promise<{ bars: BarPoint[] }> {
    return this.post("/gateway/history/bars", {
      contractId,
      timeframe,
      limit,
    });
  }

  placeOrder(payload: OrderTicketPayload): Promise<{ orderId: string }> {
    return this.post("/gateway/orders/place", payload);
  }
}

export function createGatewayClient(apiKey: string, baseUrl?: string): GatewayClient {
  const root = baseUrl ?? import.meta.env.VITE_TOPTEK_API_BASE ?? DEFAULT_BASE;
  return new GatewayClient(apiKey, root);
}
