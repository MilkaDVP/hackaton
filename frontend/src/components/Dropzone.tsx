import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";
import { Button } from "./ui";

export function Dropzone({
  onFile, onDemo, busy,
}: {
  onFile: (f: File) => void;
  onDemo: () => void;
  busy: boolean;
}) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    multiple: false,
    disabled: busy,
    accept: {
      "text/csv": [".csv", ".tsv"],
      "text/tab-separated-values": [".tsv"],
      "application/vnd.ms-excel": [".xls"],
      "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": [".xlsx"],
    },
    onDrop: (files) => files[0] && onFile(files[0]),
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        "rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors",
        busy && "pointer-events-none opacity-60",
        isDragActive
          ? "border-accent bg-accent-soft"
          : "border-line bg-surface hover:border-faint",
      )}
    >
      <input {...getInputProps()} aria-label="Файл со списком студентов" />
      <p className="text-sm font-medium">
        {isDragActive ? "Отпустите файл" : "Перетащите файл со списком студентов"}
      </p>
      <p className="mt-1 text-xs text-muted">
        CSV, TSV или XLSX. Разделитель определяется автоматически — «;» и «,» оба подойдут.
      </p>
      <div className="mt-4 flex flex-wrap items-center justify-center gap-2">
        <Button variant="primary" size="sm" disabled={busy}>
          Выбрать файл
        </Button>
        <Button
          size="sm"
          disabled={busy}
          onClick={(e) => {
            e.stopPropagation();
            onDemo();
          }}
        >
          Попробовать на демо-данных
        </Button>
      </div>
      <p className="mt-4 text-2xs text-faint">
        Файл обрабатывается в памяти и никуда не сохраняется.
      </p>
    </div>
  );
}
