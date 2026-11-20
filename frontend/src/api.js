async function j(res) {
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || "Something went wrong. Is the backend running?");
  return data;
}
export const getHealth = () => fetch("/api/health").then(j);
export const getStats = () => fetch("/api/stats").then(j);
export const getHistory = () => fetch("/api/history").then(j);
export const getSamples = () => fetch("/api/samples").then(j);
export const predictFile = (file) => {
  const fd = new FormData();
  fd.append("image", file);
  return fetch("/api/predict", { method: "POST", body: fd }).then(j);
};
export const predictSample = (name) => fetch(`/api/predict-sample/${encodeURIComponent(name)}`, { method: "POST" }).then(j);
export const reportUrl = (id) => `/api/report/${id}`;
