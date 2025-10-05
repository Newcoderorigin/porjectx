import { FormEvent, useEffect, useMemo, useState } from "react";
import {
  AccountSummary,
  BarPoint,
  ContractSummary,
  OrderTicketPayload,
  createGatewayClient,
} from "./api";
import { MarketsTable } from "./components/MarketsTable";
import { PriceChart } from "./components/PriceChart";
import { OrderTicket } from "./components/OrderTicket";
import "./styles.css";

const STORAGE_KEY = "toptek.web.api";
const DEFAULT_QUERY = "ES";
const DEFAULT_TIMEFRAME = "5m";

interface PersistedSettings {
  apiKey: string;
  baseUrl?: string;
}

function loadSettings(): PersistedSettings {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (!stored) {
      return { apiKey: "" };
    }
    return JSON.parse(stored) as PersistedSettings;
  } catch {
    return { apiKey: "" };
  }
}

export default function App() {
  const initial = loadSettings();
  const [apiKey, setApiKey] = useState(initial.apiKey);
  const [baseUrl, setBaseUrl] = useState(initial.baseUrl ?? "");
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [contracts, setContracts] = useState<ContractSummary[]>([]);
  const [contractsLoading, setContractsLoading] = useState(false);
  const [contractsError, setContractsError] = useState<string | null>(null);
  const [selectedContract, setSelectedContract] = useState<ContractSummary | null>(null);
  const [bars, setBars] = useState<BarPoint[]>([]);
  const [barsLoading, setBarsLoading] = useState(false);
  const [barsError, setBarsError] = useState<string | null>(null);
  const [accounts, setAccounts] = useState<AccountSummary[]>([]);
  const [accountsLoading, setAccountsLoading] = useState(false);
  const [accountsError, setAccountsError] = useState<string | null>(null);

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ apiKey, baseUrl }));
  }, [apiKey, baseUrl]);

  const client = useMemo(() => {
    if (!apiKey) {
      return null;
    }
    return createGatewayClient(apiKey, baseUrl || undefined);
  }, [apiKey, baseUrl]);

  useEffect(() => {
    if (!client) {
      setContracts([]);
      setSelectedContract(null);
      return;
    }
    let active = true;
    async function run() {
      setContractsLoading(true);
      setContractsError(null);
      try {
        const response = await client.searchContracts(query);
        if (!active) {
          return;
        }
        setContracts(response.items);
        if (!selectedContract && response.items.length) {
          setSelectedContract(response.items[0]);
        }
      } catch (error) {
        if (!active) {
          return;
        }
        setContractsError(error instanceof Error ? error.message : String(error));
        setContracts([]);
      } finally {
        if (active) {
          setContractsLoading(false);
        }
      }
    }
    run();
    return () => {
      active = false;
    };
  }, [client, query]);

  useEffect(() => {
    if (!client || !selectedContract) {
      setBars([]);
      return;
    }
    let active = true;
    async function run() {
      setBarsLoading(true);
      setBarsError(null);
      try {
        const response = await client.loadBars(selectedContract.contractId, DEFAULT_TIMEFRAME, 120);
        if (!active) {
          return;
        }
        setBars(response.bars);
      } catch (error) {
        if (!active) {
          return;
        }
        setBarsError(error instanceof Error ? error.message : String(error));
        setBars([]);
      } finally {
        if (active) {
          setBarsLoading(false);
        }
      }
    }
    run();
    return () => {
      active = false;
    };
  }, [client, selectedContract]);

  useEffect(() => {
    if (!client) {
      setAccounts([]);
      return;
    }
    let active = true;
    async function run() {
      setAccountsLoading(true);
      setAccountsError(null);
      try {
        const response = await client.searchAccounts();
        if (!active) {
          return;
        }
        setAccounts(response.items);
      } catch (error) {
        if (!active) {
          return;
        }
        setAccountsError(error instanceof Error ? error.message : String(error));
        setAccounts([]);
      } finally {
        if (active) {
          setAccountsLoading(false);
        }
      }
    }
    run();
    return () => {
      active = false;
    };
  }, [client]);

  async function handleSubmitOrder(payload: OrderTicketPayload) {
    if (!client) {
      throw new Error("API key required");
    }
    await client.placeOrder(payload);
  }

  function handleCredentials(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const form = event.currentTarget;
    const formData = new FormData(form);
    setApiKey((formData.get("apiKey") as string) ?? "");
    setBaseUrl(((formData.get("baseUrl") as string) ?? "").trim());
  }

  return (
    <div className="app">
      <header>
        <div>
          <h1>Toptek Markets Console</h1>
          <p>Search CME futures, review recent price action, and queue manual orders.</p>
        </div>
        <form className="credentials" onSubmit={handleCredentials}>
          <label>
            API key
            <input name="apiKey" type="password" placeholder="ProjectX API key" defaultValue={apiKey} />
          </label>
          <label>
            API base URL
            <input name="baseUrl" type="text" placeholder="https://localhost:8000" defaultValue={baseUrl} />
          </label>
          <button type="submit">Save</button>
        </form>
      </header>
      {!apiKey ? <div className="panel warning">Enter an API key to load data from the gateway.</div> : null}
      <div className="layout">
        <section>
          <div className="panel search">
            <label>
              Contract filter
              <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="ES" />
            </label>
          </div>
          <MarketsTable
            contracts={contracts}
            selectedId={selectedContract?.contractId}
            loading={contractsLoading}
            error={contractsError}
            onSelect={(contract) => setSelectedContract(contract)}
          />
          {accountsError ? <div className="panel error">{accountsError}</div> : null}
          {accountsLoading ? <div className="panel">Loading accounts…</div> : null}
        </section>
        <main>
          <PriceChart data={bars} loading={barsLoading} error={barsError} />
          <OrderTicket
            contractId={selectedContract?.contractId}
            accounts={accounts}
            onSubmit={handleSubmitOrder}
          />
        </main>
      </div>
    </div>
  );
}
