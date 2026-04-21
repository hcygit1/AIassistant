"use client";

type PipixiaMarkProps = {
  className?: string;
  strokeWidth?: number;
};

export default function PipixiaMark({
  className = "h-4 w-4",
  strokeWidth = 1.8,
}: PipixiaMarkProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className={className}
      aria-hidden="true"
    >
      <path
        d="M6.5 9.8c1.6-2.3 4.2-3.8 7.6-3.5 2.4.2 4.3 1.4 5.5 3.3"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M6.6 10c.2 1.8 1.1 3.3 2.6 4.4 1.3.9 2.7 1.5 4.1 1.8"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M18.9 10.3c-.4 1.7-1.5 3.2-3.2 4.1"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M5.8 10 3.6 7.9"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M5.2 11.8 2.5 11.6"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M5.7 13.5 3.9 15.3"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M12.5 15.9c.4 1 .3 2-.4 2.9"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M14.7 15.2c.6.9.6 1.9.1 2.9"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <path
        d="M16.8 14c.8.7 1.2 1.7 1 2.9"
        stroke="currentColor"
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
      <circle cx="15.8" cy="8.3" r="0.9" fill="currentColor" />
    </svg>
  );
}
