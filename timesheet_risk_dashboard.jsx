import React, { useMemo, useState } from "react";
import { ClipboardList, CheckCircle2, AlertTriangle, Info } from "lucide-react";

// ---- Exported from the trained scikit-learn Logistic Regression model ----
// (models/logreg_coefficients.json in the project repo)
const MODEL = {
  features: [
    "caregiver_tenure_months", "scheduled_duration_min", "duration_deviation_min",
    "arrival_deviation_min", "tasks_assigned", "task_completion_rate",
    "has_signature", "gps_mismatch_km", "is_weekend", "entry_method_paper",
  ],
  coefficients: [
    -0.12387756595510695, 0.06596364027264053, -0.042833963321827324,
    0.03467708348437259, -0.041436092114469664, -1.226529186938718,
    -0.8200434718854723, 0.2105710025906017, 0.048736303343343716,
    0.5337415621781965,
  ],
  intercept: -0.8187753610557166,
  scaler_mean: [
    29.590888888888887, 72.15333333333334, 0.16044444444444445, 0.1668888888888889,
    4.0055555555555555, 0.6974006666666667, 0.9513333333333334, 0.3571288888888889,
    0.2822222222222222, 0.25266666666666665,
  ],
  scaler_scale: [
    16.77121758280765, 27.34403936672285, 9.06389858616296, 7.6564521294054035,
    1.4008617277400768, 0.14091379129571857, 0.21517021685684617, 0.3747965685970174,
    0.4500809255080389, 0.43454139299061284,
  ],
};

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function scoreVisit(raw) {
  const z = MODEL.features.map((f, i) => (raw[f] - MODEL.scaler_mean[i]) / MODEL.scaler_scale[i]);
  const contributions = z.map((zi, i) => zi * MODEL.coefficients[i]);
  const logit = contributions.reduce((a, b) => a + b, MODEL.intercept);
  const risk = sigmoid(logit);
  return { risk, contributions };
}

const FEATURE_LABELS = {
  caregiver_tenure_months: "Caregiver tenure",
  scheduled_duration_min: "Scheduled duration",
  duration_deviation_min: "Duration vs. scheduled",
  arrival_deviation_min: "Arrival time deviation",
  tasks_assigned: "Tasks assigned",
  task_completion_rate: "Task completion rate",
  has_signature: "Missing signature",
  gps_mismatch_km: "GPS location mismatch",
  is_weekend: "Weekend visit",
  entry_method_paper: "Paper entry (vs. app)",
};

function Slider({ label, value, min, max, step = 1, unit = "", onChange, hint }) {
  return (
    <div className="mb-5">
      <div className="flex items-baseline justify-between mb-1">
        <label className="text-sm font-medium text-slate-700">{label}</label>
        <span className="text-sm tabular-nums text-slate-500">
          {value}
          {unit}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-teal-700"
      />
      {hint && <p className="text-xs text-slate-400 mt-1">{hint}</p>}
    </div>
  );
}

function Toggle({ label, checked, onChange, hint }) {
  return (
    <div className="mb-5 flex items-start justify-between">
      <div>
        <p className="text-sm font-medium text-slate-700">{label}</p>
        {hint && <p className="text-xs text-slate-400 mt-0.5">{hint}</p>}
      </div>
      <button
        onClick={() => onChange(!checked)}
        className={`w-11 h-6 rounded-full relative transition-colors shrink-0 ${
          checked ? "bg-teal-700" : "bg-slate-300"
        }`}
      >
        <span
          className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
            checked ? "translate-x-5" : ""
          }`}
        />
      </button>
    </div>
  );
}

export default function TimesheetRiskDashboard() {
  const [tenure, setTenure] = useState(18);
  const [scheduledDuration, setScheduledDuration] = useState(60);
  const [actualDuration, setActualDuration] = useState(58);
  const [arrivalDeviation, setArrivalDeviation] = useState(4);
  const [tasksAssigned, setTasksAssigned] = useState(4);
  const [tasksCompleted, setTasksCompleted] = useState(4);
  const [hasSignature, setHasSignature] = useState(true);
  const [entryPaper, setEntryPaper] = useState(false);
  const [gpsMismatch, setGpsMismatch] = useState(0.2);
  const [isWeekend, setIsWeekend] = useState(false);

  const completedClamped = Math.min(tasksCompleted, tasksAssigned);
  const completionRate = tasksAssigned > 0 ? completedClamped / tasksAssigned : 0;
  const durationDeviation = actualDuration - scheduledDuration;

  const raw = {
    caregiver_tenure_months: tenure,
    scheduled_duration_min: scheduledDuration,
    duration_deviation_min: durationDeviation,
    arrival_deviation_min: arrivalDeviation,
    tasks_assigned: tasksAssigned,
    task_completion_rate: completionRate,
    has_signature: hasSignature ? 1 : 0,
    gps_mismatch_km: entryPaper ? 0 : gpsMismatch,
    is_weekend: isWeekend ? 1 : 0,
    entry_method_paper: entryPaper ? 1 : 0,
  };

  const { risk, contributions } = useMemo(() => scoreVisit(raw), [
    tenure, scheduledDuration, durationDeviation, arrivalDeviation,
    tasksAssigned, completionRate, hasSignature, entryPaper, gpsMismatch, isWeekend,
  ]);

  const riskPct = Math.round(risk * 100);
  const flagged = risk >= 0.5;

  const topFactors = MODEL.features
    .map((f, i) => ({ label: FEATURE_LABELS[f], value: contributions[i] }))
    .filter((c) => c.value > 0)
    .sort((a, b) => b.value - a.value)
    .slice(0, 3);

  return (
    <div className="min-h-screen bg-[#F6F7F5] text-slate-800 p-6 md:p-10">
      <div className="max-w-5xl mx-auto">
        <div className="flex items-center gap-3 mb-1">
          <ClipboardList className="text-teal-800" size={28} />
          <h1 className="text-2xl md:text-3xl font-semibold text-slate-900">
            Timesheet Review Risk Checker
          </h1>
        </div>
        <p className="text-sm text-slate-500 mb-8 max-w-2xl">
          Predicts whether a home-care visit timesheet is likely to need manual
          review before payroll/billing, based on a logistic regression model
          trained on simulated visit data.
        </p>

        <div className="grid md:grid-cols-5 gap-6">
          {/* Input form */}
          <div className="md:col-span-3 bg-white rounded-xl border border-slate-200 p-6">
            <h2 className="text-sm font-semibold text-slate-500 mb-4">
              Visit details
            </h2>

            <Slider
              label="Caregiver tenure"
              value={tenure}
              min={1}
              max={60}
              unit=" mo"
              onChange={setTenure}
            />
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Scheduled duration"
                value={scheduledDuration}
                min={30}
                max={120}
                step={15}
                unit=" min"
                onChange={setScheduledDuration}
              />
              <Slider
                label="Actual duration"
                value={actualDuration}
                min={10}
                max={150}
                unit=" min"
                onChange={setActualDuration}
              />
            </div>
            <Slider
              label="Arrival time deviation"
              value={arrivalDeviation}
              min={-30}
              max={30}
              unit=" min"
              onChange={setArrivalDeviation}
              hint="Negative = early, positive = late"
            />
            <div className="grid grid-cols-2 gap-4">
              <Slider
                label="Tasks assigned"
                value={tasksAssigned}
                min={2}
                max={7}
                onChange={setTasksAssigned}
              />
              <Slider
                label="Tasks completed"
                value={completedClamped}
                min={0}
                max={tasksAssigned}
                onChange={setTasksCompleted}
              />
            </div>

            <div className="border-t border-slate-100 my-4" />

            <Toggle
              label="Signature on file"
              checked={hasSignature}
              onChange={setHasSignature}
            />
            <Toggle
              label="Paper entry"
              checked={entryPaper}
              onChange={setEntryPaper}
              hint="Off = submitted via mobile app"
            />
            {!entryPaper && (
              <Slider
                label="GPS location mismatch"
                value={gpsMismatch}
                min={0}
                max={2}
                step={0.1}
                unit=" km"
                onChange={setGpsMismatch}
              />
            )}
            <Toggle
              label="Weekend visit"
              checked={isWeekend}
              onChange={setIsWeekend}
            />
          </div>

          {/* Result panel */}
          <div className="md:col-span-2 space-y-4">
            <div className="bg-white rounded-xl border border-slate-200 p-6 text-center">
              <p className="text-xs font-semibold text-slate-400 mb-2">
                PREDICTED REVIEW RISK
              </p>
              <p
                className={`text-5xl font-bold tabular-nums ${
                  flagged ? "text-amber-600" : "text-teal-700"
                }`}
              >
                {riskPct}%
              </p>
              <div
                className={`mt-4 inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium ${
                  flagged
                    ? "bg-amber-50 text-amber-700"
                    : "bg-teal-50 text-teal-700"
                }`}
              >
                {flagged ? (
                  <AlertTriangle size={16} />
                ) : (
                  <CheckCircle2 size={16} />
                )}
                {flagged ? "Send to manual review" : "Auto-approve"}
              </div>
            </div>

            <div className="bg-white rounded-xl border border-slate-200 p-6">
              <p className="text-xs font-semibold text-slate-400 mb-3">
                TOP RISK FACTORS
              </p>
              {topFactors.length === 0 ? (
                <p className="text-sm text-slate-400">
                  No significant risk factors detected.
                </p>
              ) : (
                <ul className="space-y-2">
                  {topFactors.map((f) => (
                    <li
                      key={f.label}
                      className="flex items-center justify-between text-sm"
                    >
                      <span className="text-slate-600">{f.label}</span>
                      <span className="text-slate-400 tabular-nums">
                        +{f.value.toFixed(2)}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </div>

            <div className="bg-slate-50 rounded-xl border border-slate-200 p-4 flex gap-2">
              <Info size={16} className="text-slate-400 shrink-0 mt-0.5" />
              <p className="text-xs text-slate-500 leading-relaxed">
                Demo uses simulated data modeled on home-care visit
                verification. Model: logistic regression, ROC-AUC 0.84 on held-out
                data (see repo for full comparison against Random Forest, AUC 0.88).
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
