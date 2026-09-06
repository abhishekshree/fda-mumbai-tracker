import { unstable_cache } from "next/cache";

import { sql } from "./db";
import { activeFilter } from "./href";
import type { ActionRow, DimCount, Filters, RunSummary, Stats } from "./types";

const DAILY = { revalidate: 21600 };

const toInt = (v: unknown): number =>
  typeof v === "string" ? parseInt(v, 10) : Number(v ?? 0);

async function queryRows<T extends Record<string, unknown>>(
  text: string,
  params: unknown[] = [],
): Promise<T[]> {
  return (await sql(text, params)) as T[];
}

function normalizeAction(r: Record<string, unknown>): ActionRow {
  return {
    id: toInt(r.id),
    establishment: String(r.establishment ?? ""),
    area: r.area ? String(r.area) : null,
    city: r.city ? String(r.city) : null,
    state: String(r.state ?? ""),
    brand: r.brand ? String(r.brand) : null,
    operator: r.operator ? String(r.operator) : null,
    outlet_type: r.outlet_type ? String(r.outlet_type) : null,
    action_type: String(r.action_type ?? ""),
    action_date: String(r.action_date ?? ""),
    status: String(r.status ?? ""),
    violations: Array.isArray(r.violations) ? r.violations.map(String) : [],
    compliance_score:
      r.compliance_score === null || r.compliance_score === undefined
        ? null
        : toInt(r.compliance_score),
    fssai_number: r.fssai_number ? String(r.fssai_number) : null,
    details: r.details ? String(r.details) : null,
    platforms: Array.isArray(r.platforms) ? r.platforms.map(String) : [],
    source_url: String(r.source_url ?? ""),
    source_publisher: r.source_publisher ? String(r.source_publisher) : null,
    source_headline: r.source_headline ? String(r.source_headline) : null,
    published_at: r.published_at ? String(r.published_at) : null,
    created_at: String(r.created_at ?? ""),
    updated_at: String(r.updated_at ?? ""),
  };
}

export const listActions = unstable_cache(
  async (
    f: Filters,
    opts: { limit: number; offset: number },
  ): Promise<{ total: number; actions: ActionRow[] }> => {
    const params: unknown[] = [];
    const conds: string[] = [];

    const eq = (col: string, v: string | undefined) => {
      const val = activeFilter(v);
      if (val) {
        params.push(val);
        conds.push(`${col} = $${params.length}`);
      }
    };

    eq("brand", f.brand);
    eq("city", f.city);
    eq("status", f.status);
    eq("action_type", f.action_type);
    eq("outlet_type", f.outlet_type);

    const q = f.q?.trim();
    if (q) {
      const like = `%${q}%`;
      params.push(like, like, like);
      conds.push(
        `(establishment ILIKE $${params.length - 2} OR brand ILIKE $${params.length - 1} OR area ILIKE $${params.length})`,
      );
    }
    if (f.from) {
      params.push(f.from);
      conds.push(`action_date >= $${params.length}`);
    }
    if (f.to) {
      params.push(f.to);
      conds.push(`action_date <= $${params.length}`);
    }

    const where = conds.length ? `WHERE ${conds.join(" AND ")}` : "WHERE true";
    const filterParams = [...params];
    const limit = Math.min(Math.max(opts.limit, 1), 200);
    const offset = Math.max(Math.floor(opts.offset), 0);
    const text = `SELECT actions.*, COUNT(*) OVER()::int AS total_count FROM actions ${where}
      ORDER BY action_date DESC, id DESC LIMIT $${filterParams.length + 1} OFFSET $${filterParams.length + 2}`;

    const rows = (await sql(text, [...filterParams, limit, offset])) as Array<
      Record<string, unknown>
    >;
    const total = rows[0]
      ? toInt(rows[0].total_count)
      : offset
        ? toInt(
            (
              (await sql(
                `SELECT COUNT(*)::int AS total_count FROM actions ${where}`,
                filterParams,
              )) as Array<Record<string, unknown>>
            )[0]?.total_count,
          )
        : 0;
    return {
      total,
      actions: rows.map(normalizeAction),
    };
  },
  ["list-actions", "v2"],
  DAILY,
);

async function dims(col: string, limit?: number): Promise<DimCount[]> {
  const query = `SELECT COALESCE(NULLIF(${col}, ''), 'unknown') AS k, COUNT(*)::int AS n
     FROM actions GROUP BY k ORDER BY n DESC${limit ? " LIMIT $1" : ""}`;
  const rows = (await sql(
    query,
    limit ? [limit] : [],
  )) as Array<Record<string, unknown>>;
  return rows.map((r) => ({
    key: String(r.k),
    n: toInt(r.n),
  }));
}

export const listBrands = unstable_cache(
  async (): Promise<string[]> => {
    const rows = (await sql(
      "SELECT DISTINCT brand FROM actions WHERE brand IS NOT NULL AND brand <> '' ORDER BY brand",
    )) as Array<Record<string, unknown>>;
    return rows.map((r) => String(r.brand));
  },
  ["list-brands", "v2"],
  DAILY,
);

export const listCities = unstable_cache(
  async (): Promise<string[]> => {
    const rows = (await sql(
      "SELECT DISTINCT city FROM actions WHERE city IS NOT NULL AND city <> '' ORDER BY city",
    )) as Array<Record<string, unknown>>;
    return rows.map((r) => String(r.city));
  },
  ["list-cities", "v2"],
  DAILY,
);

function normalizeRun(r: Record<string, unknown>): RunSummary {
  return {
    id: toInt(r.id),
    started_at: String(r.started_at ?? ""),
    finished_at: r.finished_at ? String(r.finished_at) : null,
    status: String(r.status ?? ""),
    articles_seen: toInt(r.articles_seen),
    articles_new: toInt(r.articles_new),
    actions_upserted: toInt(r.actions_upserted),
    llm_calls: toInt(r.llm_calls),
    error: r.error ? String(r.error) : null,
  };
}

export const getStats = unstable_cache(
  async (): Promise<Stats> => {
    const [total, runs, byStatus, byActionType, byCity] = await Promise.all([
      queryRows<{ n: unknown }>("SELECT COUNT(*)::int AS n FROM actions"),
      queryRows<Record<string, unknown>>(`SELECT id::int AS id, started_at, finished_at, status,
            articles_seen::int AS articles_seen, articles_new::int AS articles_new,
            actions_upserted::int AS actions_upserted, llm_calls::int AS llm_calls, error
     FROM fetch_runs ORDER BY id DESC LIMIT 1`),
      dims("status"),
      dims("action_type"),
      dims("city", 8),
    ]);

    return {
      totalActions: total[0] ? toInt(total[0].n) : 0,
      lastRun: runs[0] ? normalizeRun(runs[0]) : null,
      byStatus,
      byActionType,
      byCity,
    };
  },
  ["get-stats", "v2"],
  DAILY,
);
