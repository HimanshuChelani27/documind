import { Routes } from '@angular/router';

import { Shell } from './layout/shell';

export const routes: Routes = [
  {
    path: '',
    component: Shell,
    children: [
      { path: '', pathMatch: 'full', redirectTo: 'chat' },
      { path: 'chat', loadComponent: () => import('./pages/chat').then((m) => m.Chat), title: 'Ask · DocuMind' },
      {
        path: 'documents',
        loadComponent: () => import('./pages/documents').then((m) => m.Documents),
        title: 'Documents · DocuMind',
      },
      {
        path: 'documents/:id',
        loadComponent: () => import('./pages/document-detail').then((m) => m.DocumentDetail),
        title: 'Chunks · DocuMind',
      },
      {
        path: 'evaluate',
        loadComponent: () => import('./pages/evaluate').then((m) => m.Evaluate),
        title: 'Evaluate · DocuMind',
      },
    ],
  },
  { path: '**', redirectTo: '' },
];
