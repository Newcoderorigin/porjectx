import { ContractSummary } from "../api";

export interface MarketsTableProps {
  contracts: ContractSummary[];
  selectedId?: string;
  loading?: boolean;
  error?: string | null;
  onSelect(contract: ContractSummary): void;
}

export function MarketsTable({
  contracts,
  selectedId,
  loading = false,
  error = null,
  onSelect,
}: MarketsTableProps) {
  if (loading) {
    return <div className="panel">Loading contracts…</div>;
  }
  if (error) {
    return <div className="panel error">{error}</div>;
  }
  if (!contracts.length) {
    return <div className="panel">No contracts match your filter.</div>;
  }
  return (
    <div className="panel">
      <table className="markets-table">
        <thead>
          <tr>
            <th>Symbol</th>
            <th>Description</th>
            <th>Tick Size</th>
          </tr>
        </thead>
        <tbody>
          {contracts.map((contract) => (
            <tr
              key={contract.contractId}
              className={contract.contractId === selectedId ? "selected" : undefined}
              onClick={() => onSelect(contract)}
            >
              <td>{contract.symbol}</td>
              <td>{contract.description}</td>
              <td>{contract.tickSize}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
