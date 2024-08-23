import { Routes } from '@angular/router';
import { CourierDetailsComponent } from './courier-details/courier-details.component';
import { FirstPageComponent } from './first-page/first-page.component';
import { PriceComponent } from './price/price.component';
import { SchemaComponent } from './schema/schema.component';

export const routes: Routes = [{ path: '', redirectTo: '/home', pathMatch: 'full' },
{ path: 'home', component:FirstPageComponent},
{ path: 'with-courier', component: CourierDetailsComponent },
{ path: 'price', component: PriceComponent },
{ path: 'schema', component: SchemaComponent },

];
