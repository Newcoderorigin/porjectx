import { FormEvent, useEffect, useState } from "react";
import { OrderTicketPayload } from "../api";

export interface OrderTicketProps {
  contractId?: string;
  accounts: { accountId: string; name: string }[];
  onSubmit(payload: OrderTicketPayload): Promise<void>;
}

export function OrderTicket({ contractId, accounts, onSubmit }: OrderTicketProps) {
  const [accountId, setAccountId] = useState(accounts[0]?.accountId ?? "");
  const [quantity, setQuantity] = useState(1);
  const [side, setSide] = useState<OrderTicketPayload["side"]>("BUY");
  const [orderType, setOrderType] = useState<OrderTicketPayload["orderType"]>("MARKET");
  const [limitPrice, setLimitPrice] = useState(0);
  const [status, setStatus] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const disabled = !contractId || !accountId || submitting;

  useEffect(() => {
    if (!accounts.length) {
      setAccountId("");
      return;
    }
    if (!accountId || !accounts.some((item) => item.accountId === accountId)) {
      setAccountId(accounts[0].accountId);
    }
  }, [accounts]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (disabled || !contractId) {
      return;
    }
    setSubmitting(true);
    setStatus(null);
    try {
      await onSubmit({
        accountId,
        contractId,
        quantity,
        side,
        orderType,
        price: orderType === "LIMIT" ? limitPrice : undefined,
      });
      setStatus("Order submitted successfully");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : String(error));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <form className="panel order-ticket" onSubmit={handleSubmit}>
      <h3>Order Ticket</h3>
      <label>
        Account
        <select value={accountId} onChange={(event) => setAccountId(event.target.value)}>
          {accounts.map((account) => (
            <option key={account.accountId} value={account.accountId}>
              {account.name}
            </option>
          ))}
        </select>
      </label>
      <label>
        Quantity
        <input type="number" min={1} value={quantity} onChange={(event) => setQuantity(Number(event.target.value))} />
      </label>
      <label className="radio-group">
        Side
        <div>
          <label>
            <input type="radio" name="side" value="BUY" checked={side === "BUY"} onChange={() => setSide("BUY")} /> Buy
          </label>
          <label>
            <input type="radio" name="side" value="SELL" checked={side === "SELL"} onChange={() => setSide("SELL")} /> Sell
          </label>
        </div>
      </label>
      <label>
        Order Type
        <select value={orderType} onChange={(event) => setOrderType(event.target.value as OrderTicketPayload["orderType"])}>
          <option value="MARKET">Market</option>
          <option value="LIMIT">Limit</option>
        </select>
      </label>
      {orderType === "LIMIT" ? (
        <label>
          Limit Price
          <input type="number" value={limitPrice} onChange={(event) => setLimitPrice(Number(event.target.value))} />
        </label>
      ) : null}
      <button type="submit" disabled={disabled}>
        {submitting ? "Submitting…" : "Submit Order"}
      </button>
      {status ? <p className="status">{status}</p> : null}
    </form>
  );
}
