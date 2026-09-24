{{- define "inference-lab-dynamo.namespaces" -}}
{{- if .Values.namespaces -}}
{{- toJson .Values.namespaces -}}
{{- else -}}
{{- toJson (list .Release.Namespace) -}}
{{- end -}}
{{- end -}}

{{- define "inference-lab-dynamo.policyName" -}}
{{- printf "%s-%s" .Release.Name .Release.Namespace | trunc 63 | trimSuffix "-" -}}
{{- end -}}
